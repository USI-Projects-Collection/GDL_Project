import React, { useState, useRef, useEffect, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";

// --- CONFIGURAZIONE ---
const NUM_POINTS = 2048;         
const RECORD_INTERVAL = 0.1;   // 10 FPS
const K_NEIGHBORS = 6;

// Parametri del braccio
const ARM_SEGMENTS = {
  base: { height: 0.2, radius: 0.15 },
  link1: { length: 0.6, radius: 0.08 },
  link2: { length: 0.5, radius: 0.06 },
  wrist: { length: 0.1, radius: 0.07 },
  finger: { length: 0.15, radius: 0.02 }
};

// --- HELPER: NPY Parser ---
function parseNpy(buffer: ArrayBuffer): { data: Float32Array, shape: number[] } {
  const headerLenView = new DataView(buffer.slice(8, 10));
  const headerLen = headerLenView.getUint16(0, true);
  const headerStr = new TextDecoder("ascii").decode(buffer.slice(10, 10 + headerLen));
  
  const shapeMatch = headerStr.match(/'shape': \((.*?)\)/);
  if (!shapeMatch) throw new Error("Invalid NPY header: shape not found");
  
  const shape = shapeMatch[1].split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const data = new Float32Array(buffer.slice(10 + headerLen));
  
  return { data, shape };
}

// --- HELPER: NPY Writer ---
function createNpyHeader(shape: number[], dtype: string): Uint8Array {
  const magic = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]);
  const version = new Uint8Array([1, 0]); 
  let shapeStr = shape.join(", ");
  if (shape.length === 1) shapeStr += ","; 
  let headerStr = `{ 'descr': '${dtype}', 'fortran_order': False, 'shape': (${shapeStr}), }`;
  const currentLen = 10 + headerStr.length;
  const paddingNeeded = 64 - (currentLen % 64);
  headerStr += " ".repeat(paddingNeeded - 1) + "\n";
  const headerBytes = new TextEncoder().encode(headerStr);
  const headerLenVal = headerBytes.length;
  const headerLen = new Uint8Array([headerLenVal & 0xFF, (headerLenVal >> 8) & 0xFF]);
  const fullHeader = new Uint8Array(10 + headerBytes.length);
  fullHeader.set(magic, 0);
  fullHeader.set(version, 6);
  fullHeader.set(headerLen, 8);
  fullHeader.set(headerBytes, 10);
  return fullHeader;
}

function serializeNpy(data: Float32Array | Uint32Array, shape: number[], dtype: string): Blob {
  const header = createNpyHeader(shape, dtype);
  const body = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Blob([header, body], { type: "application/octet-stream" });
}

// --- HELPER: Geometria ---
function sampleCylinder(radius: number, height: number, numPoints: number, transform: THREE.Matrix4): Float32Array {
  const points = new Float32Array(numPoints * 3);
  const vec = new THREE.Vector3();
  for (let i = 0; i < numPoints; i++) {
    const theta = Math.random() * Math.PI * 2;
    const y = (Math.random() - 0.5) * height;
    vec.set(Math.cos(theta) * radius, y, Math.sin(theta) * radius);
    vec.applyMatrix4(transform);
    points[i * 3] = vec.x; points[i * 3 + 1] = vec.y; points[i * 3 + 2] = vec.z;
  }
  return points;
}

function sampleBox(width: number, height: number, depth: number, numPoints: number, transform: THREE.Matrix4): Float32Array {
  const points = new Float32Array(numPoints * 3);
  const vec = new THREE.Vector3();
  for (let i = 0; i < numPoints; i++) {
    vec.set((Math.random() - 0.5) * width, (Math.random() - 0.5) * height, (Math.random() - 0.5) * depth);
    vec.applyMatrix4(transform);
    points[i * 3] = vec.x; points[i * 3 + 1] = vec.y; points[i * 3 + 2] = vec.z;
  }
  return points;
}

function computeKNN(positions: Float32Array, k: number): Uint32Array {
  const n = positions.length / 3;
  const indices = new Uint32Array(n * k);
  const dists = new Float32Array(n);
  const neighborIndices = new Int32Array(n); 
  for (let i = 0; i < n; i++) {
    const x1 = positions[i * 3]; const y1 = positions[i * 3 + 1]; const z1 = positions[i * 3 + 2];
    for (let j = 0; j < n; j++) {
      const dx = x1 - positions[j * 3]; const dy = y1 - positions[j * 3 + 1]; const dz = z1 - positions[j * 3 + 2];
      dists[j] = dx*dx + dy*dy + dz*dz; neighborIndices[j] = j;
    }
    neighborIndices.sort((a, b) => dists[a] - dists[b]);
    for (let j = 0; j < k; j++) { indices[i * k + j] = neighborIndices[j + 1]; }
  }
  return indices;
}

// --- COMPONENTE 1: SIMULATORE (Generazione Dati) ---
const RobotSimulation = ({ timeRef, isPlaying, showGraph, showColors, onUpdate, resetTime }: any) => {
  const pointsRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  
  const structure = useMemo(() => {
    const totalPoints = NUM_POINTS;
    const dist = [0.15, 0.35, 0.35, 0.15]; 
    const counts = dist.map(p => Math.floor(p * totalPoints));
    counts[3] += totalPoints - counts.reduce((a,b) => a+b, 0);
    const baseLocal = sampleBox(0.3, 0.2, 0.3, counts[0], new THREE.Matrix4());
    const l1Local = sampleCylinder(ARM_SEGMENTS.link1.radius, ARM_SEGMENTS.link1.length, counts[1], new THREE.Matrix4());
    const l2Local = sampleCylinder(ARM_SEGMENTS.link2.radius, ARM_SEGMENTS.link2.length, counts[2], new THREE.Matrix4());
    const fingerPointsPerFinger = Math.floor(counts[3] / 3);
    const f1Local = sampleCylinder(ARM_SEGMENTS.finger.radius, ARM_SEGMENTS.finger.length, fingerPointsPerFinger, new THREE.Matrix4());
    const f2Local = sampleCylinder(ARM_SEGMENTS.finger.radius, ARM_SEGMENTS.finger.length, fingerPointsPerFinger, new THREE.Matrix4());
    const f3Local = sampleCylinder(ARM_SEGMENTS.finger.radius, ARM_SEGMENTS.finger.length, counts[3] - 2*fingerPointsPerFinger, new THREE.Matrix4());
    return { counts, localPoints: [baseLocal, l1Local, l2Local, f1Local, f2Local, f3Local] };
  }, []);

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(NUM_POINTS * 3);
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const colors = new Float32Array(NUM_POINTS * 3);
    const colorMap = [[0.3, 0.3, 0.3], [0.2, 0.6, 1.0], [1.0, 0.5, 0.2], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]];
    let ptr = 0;
    structure.localPoints.forEach((pts, i) => {
      const c = colorMap[i];
      for (let j=0; j<pts.length/3; j++) {
        colors[ptr*3] = c[0]; colors[ptr*3+1] = c[1]; colors[ptr*3+2] = c[2]; ptr++;
      }
    });
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geom;
  }, [structure]);

  const lineGeometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(NUM_POINTS * K_NEIGHBORS * 2 * 3); 
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    return geom;
  }, []);

  useFrame((state, delta) => {
    if (!isPlaying) return;
    
    // RESET LOGIC
    if (resetTime.current) {
        timeRef.current = 0;
        resetTime.current = false;
    }
    
    timeRef.current += delta;
    const t = timeRef.current;

    // Cinematica
    const q_base = Math.sin(t * 0.5) * 1.5;
    const q_shoulder = Math.sin(t * 0.8) * 0.8; 
    const q_elbow = Math.cos(t * 1.2) * 1.0 + 1.0; 
    const q_wrist_rot = t * 2.0;

    const mBase = new THREE.Matrix4().makeTranslation(0, 0.1, 0); 
    mBase.multiply(new THREE.Matrix4().makeRotationY(q_base));
    const mLink1 = mBase.clone();
    mLink1.multiply(new THREE.Matrix4().makeTranslation(0, 0.2, 0)); 
    mLink1.multiply(new THREE.Matrix4().makeRotationZ(q_shoulder));
    mLink1.multiply(new THREE.Matrix4().makeTranslation(0, ARM_SEGMENTS.link1.length / 2, 0)); 
    const mLink2 = mBase.clone();
    mLink2.multiply(new THREE.Matrix4().makeTranslation(0, 0.2, 0)); 
    mLink2.multiply(new THREE.Matrix4().makeRotationZ(q_shoulder));
    mLink2.multiply(new THREE.Matrix4().makeTranslation(0, ARM_SEGMENTS.link1.length, 0)); 
    mLink2.multiply(new THREE.Matrix4().makeRotationZ(q_elbow));
    mLink2.multiply(new THREE.Matrix4().makeTranslation(0, ARM_SEGMENTS.link2.length / 2, 0)); 
    const mHand = mLink2.clone();
    mHand.multiply(new THREE.Matrix4().makeTranslation(0, ARM_SEGMENTS.link2.length / 2, 0)); 
    
    const fingerTransforms = [];
    for(let i=0; i<3; i++) {
        const angle = (i / 3) * Math.PI * 2;
        const mF = mHand.clone();
        mF.multiply(new THREE.Matrix4().makeRotationY(q_wrist_rot));
        mF.multiply(new THREE.Matrix4().makeRotationZ(0.2)); 
        mF.multiply(new THREE.Matrix4().makeTranslation(Math.cos(angle)*0.05, 0.05, Math.sin(angle)*0.05));
        fingerTransforms.push(mF);
    }

    const positions = geometry.attributes.position.array as Float32Array;
    let ptr = 0;
    const apply = (localPts: Float32Array, mat: THREE.Matrix4) => {
       const v = new THREE.Vector3();
       for(let i=0; i<localPts.length/3; i++) {
         v.set(localPts[i*3], localPts[i*3+1], localPts[i*3+2]);
         v.applyMatrix4(mat);
         positions[ptr*3] = v.x; positions[ptr*3+1] = v.y; positions[ptr*3+2] = v.z; ptr++;
       }
    };
    apply(structure.localPoints[0], mBase);
    apply(structure.localPoints[1], mLink1);
    apply(structure.localPoints[2], mLink2);
    apply(structure.localPoints[3], fingerTransforms[0]);
    apply(structure.localPoints[4], fingerTransforms[1]);
    apply(structure.localPoints[5], fingerTransforms[2]);

    geometry.attributes.position.needsUpdate = true;
    
    // Calcolo KNN per visualizzazione
    if (showGraph) {
        const knnIndices = computeKNN(positions, K_NEIGHBORS);
        const linePos = lineGeometry.attributes.position.array as Float32Array;
        let linePtr = 0;
        const step = 2; 
        for(let i=0; i<NUM_POINTS; i+=step) {
           const x1 = positions[i*3]; const y1 = positions[i*3+1]; const z1 = positions[i*3+2];
           for(let k=0; k<K_NEIGHBORS; k++) {
              const neighborIdx = knnIndices[i*K_NEIGHBORS + k];
              const x2 = positions[neighborIdx*3]; const y2 = positions[neighborIdx*3+1]; const z2 = positions[neighborIdx*3+2];
              linePos[linePtr++] = x1; linePos[linePtr++] = y1; linePos[linePtr++] = z1;
              linePos[linePtr++] = x2; linePos[linePtr++] = y2; linePos[linePtr++] = z2;
           }
        }
        for(; linePtr < linePos.length; linePtr++) linePos[linePtr] = 0;
        lineGeometry.attributes.position.needsUpdate = true;
        lineGeometry.setDrawRange(0, (NUM_POINTS/step) * K_NEIGHBORS * 2);
    } else {
        lineGeometry.setDrawRange(0, 0);
    }

    if (onUpdate) onUpdate(positions, t);
  });

  return (
    <>
      <points ref={pointsRef} geometry={geometry}>
        {showColors ? <pointsMaterial key="col" size={0.03} vertexColors /> : <pointsMaterial key="mono" size={0.03} vertexColors={false} color="#ffffff" />}
      </points>
      <lineSegments ref={linesRef} geometry={lineGeometry}>
        <lineBasicMaterial key="boldLines" color="#ff4444" transparent={false} opacity={1.0} />
      </lineSegments>
    </>
  );
};

// --- COMPONENTE 2: VISUALIZZATORE PLAYBACK (Con Restart e No-Flash) ---
const PlaybackViz = ({ groundTruthSeq, predictionSeq, manualRestartTrigger }: { groundTruthSeq: {data: Float32Array, shape: number[]} | null, predictionSeq: {data: Float32Array, shape: number[]} | null, manualRestartTrigger: number }) => {
  const gtRef = useRef<THREE.Points>(null);
  const predRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  
  // State per il loop
  const playbackTimeRef = useRef(0);
  const lastTriggerRef = useRef(manualRestartTrigger);
  
  const totalFrames = groundTruthSeq ? groundTruthSeq.shape[0] : 0;
  const numPoints = groundTruthSeq ? groundTruthSeq.shape[1] : 0;

  useFrame((state, delta) => {
      if (totalFrames === 0) return;
      
      // Check Manual Restart
      if (manualRestartTrigger !== lastTriggerRef.current) {
          playbackTimeRef.current = 0;
          lastTriggerRef.current = manualRestartTrigger;
      }
      
      // Velocità playback
      playbackTimeRef.current += delta;
      const fps = 10;
      const currentFrame = Math.floor(playbackTimeRef.current * fps) % totalFrames;
      
      // Aggiorna Ground Truth
      if (gtRef.current && groundTruthSeq) {
          const start = currentFrame * numPoints * 3;
          const end = start + numPoints * 3;
          const frameData = groundTruthSeq.data.subarray(start, end);
          
          const attr = gtRef.current.geometry.attributes.position;
          // Imposta direttamente il buffer per evitare flickering
          // (attr.array as Float32Array).set(frameData);
          // Invece di .set(), se il buffer è condiviso, è meglio non riallocarlo ma scriverci sopra.
          // In Three.js, geometry.attributes.position.array è un TypedArray. 
          // .set() è veloce.
          (attr.array as Float32Array).set(frameData);
          attr.needsUpdate = true;
      }

      // Aggiorna Predizione
      let predFrameData: Float32Array | null = null;
      
      if (predRef.current && predictionSeq) {
          if (predictionSeq.shape[0] === 1 || predictionSeq.shape.length === 2) {
              predFrameData = predictionSeq.data; 
          } else {
              const safeFrame = currentFrame % predictionSeq.shape[0]; 
              const start = safeFrame * numPoints * 3;
              const end = start + numPoints * 3;
              predFrameData = predictionSeq.data.subarray(start, end);
          }

          const attr = predRef.current.geometry.attributes.position;
          (attr.array as Float32Array).set(predFrameData);
          attr.needsUpdate = true;
      }
      
      // Aggiorna Linee di Errore
      if (linesRef.current && gtRef.current && predFrameData) {
          const gtAttr = gtRef.current.geometry.attributes.position.array as Float32Array;
          const lineAttr = linesRef.current.geometry.attributes.position;
          const lineArr = lineAttr.array as Float32Array;
          
          for(let i=0; i<numPoints; i++) {
              // Start (GT)
              lineArr[i*6+0] = gtAttr[i*3+0];
              lineArr[i*6+1] = gtAttr[i*3+1];
              lineArr[i*6+2] = gtAttr[i*3+2];
              
              // End (Pred)
              lineArr[i*6+3] = predFrameData[i*3+0];
              lineArr[i*6+4] = predFrameData[i*3+1];
              lineArr[i*6+5] = predFrameData[i*3+2];
          }
          lineAttr.needsUpdate = true;
      }
  });

  const initialGeo = useMemo(() => {
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(NUM_POINTS * 3);
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      return geo;
  }, []);

  const linesGeo = useMemo(() => {
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(NUM_POINTS * 2 * 3); 
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      return geo;
  }, []);

  return (
    <group>
       <points ref={gtRef} geometry={initialGeo}>
           <pointsMaterial size={0.03} color="#ffffff" sizeAttenuation={true} />
       </points>
       {predictionSeq && (
           <points ref={predRef} geometry={initialGeo.clone()}>
               <pointsMaterial size={0.04} color="#ff0000" sizeAttenuation={true} />
           </points>
       )}
       {predictionSeq && (
           <lineSegments ref={linesRef} geometry={linesGeo}>
               <lineBasicMaterial color="#555555" transparent opacity={0.4} />
           </lineSegments>
       )}
    </group>
  );
}

// --- MAIN APP ---
const App = () => {
  const [mode, setMode] = useState<'SIM'|'PLAYBACK'>('SIM');
  
  // Sim State
  const [isPlaying, setIsPlaying] = useState(true);
  const [showGraph, setShowGraph] = useState(false);
  const [showColors, setShowColors] = useState(false);
  const timeRef = useRef(0);
  const lastRecordTimeRef = useRef(-1);
  const resetTimeRef = useRef(false);
  
  const [isRecording, setIsRecording] = useState(false);
  const [recordedData, setRecordedData] = useState<{ points: Float32Array[], knn: Uint32Array[], times: number[] }>({ points: [], knn: [], times: [] });

  // Playback State
  const [gtSeq, setGtSeq] = useState<{data: Float32Array, shape: number[]} | null>(null);
  const [predSeq, setPredSeq] = useState<{data: Float32Array, shape: number[]} | null>(null);
  const [restartTrigger, setRestartTrigger] = useState(0); // Counter to trigger restarts

  const handleUpdate = (positions: Float32Array, t: number) => {
    if (isRecording) {
       if (t - lastRecordTimeRef.current >= RECORD_INTERVAL) {
           lastRecordTimeRef.current = t;
           const posCopy = new Float32Array(positions);
           const knn = computeKNN(positions, K_NEIGHBORS);
           setRecordedData(prev => ({ points: [...prev.points, posCopy], knn: [...prev.knn, knn], times: [...prev.times, t] }));
       }
    }
  };

  const startRecording = () => {
    setRecordedData({ points: [], knn: [], times: [] });
    setIsRecording(true);
    resetTimeRef.current = true; 
    lastRecordTimeRef.current = -1;
  };

  const stopAndSave = async () => {
    setIsRecording(false);
    setIsPlaying(false);
    
    const numFrames = recordedData.times.length;
    if (numFrames === 0) { alert("No frames recorded."); return; }
    
    const pointsFlat = new Float32Array(numFrames * NUM_POINTS * 3);
    for(let i=0; i<numFrames; i++) pointsFlat.set(recordedData.points[i], i * NUM_POINTS * 3);
    const knnFlat = new Uint32Array(numFrames * NUM_POINTS * K_NEIGHBORS);
    for(let i=0; i<numFrames; i++) knnFlat.set(recordedData.knn[i], i * NUM_POINTS * K_NEIGHBORS);
    const timesFlat = new Float32Array(recordedData.times);

    const pointsBlob = serializeNpy(pointsFlat, [numFrames, NUM_POINTS, 3], '<f4');
    const knnBlob = serializeNpy(knnFlat, [numFrames, NUM_POINTS, K_NEIGHBORS], '<u4');
    const timeBlob = serializeNpy(timesFlat, [numFrames], '<f4');

    const download = (blob: Blob, name: string) => {
        const url = URL.createObjectURL(blob); const a = document.createElement('a');
        a.href = url; a.download = name; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    };

    if ('showDirectoryPicker' in window) {
      try {
        const dirHandle = await (window as any).showDirectoryPicker();
        if (!dirHandle) return;
        const pFile = await dirHandle.getFileHandle('points.npy', { create: true });
        const pW = await pFile.createWritable(); await pW.write(pointsBlob); await pW.close();
        const kFile = await dirHandle.getFileHandle('knn_indices.npy', { create: true });
        const kW = await kFile.createWritable(); await kW.write(knnBlob); await kW.close();
        const tFile = await dirHandle.getFileHandle('timestamps.npy', { create: true });
        const tW = await tFile.createWritable(); await tW.write(timeBlob); await tW.close();
        alert("Saved successfully."); return;
      } catch(e) { console.warn(e); }
    }
    download(pointsBlob, 'points.npy');
    setTimeout(() => download(knnBlob, 'knn_indices.npy'), 200);
    setTimeout(() => download(timeBlob, 'timestamps.npy'), 400);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, type: 'GT' | 'PRED') => {
      const file = e.target.files?.[0];
      if (!file) return;
      const buf = await file.arrayBuffer();
      try {
          const result = parseNpy(buf);
          console.log(`Loaded ${type}:`, result.shape);
          
          if (type === 'GT') {
              setGtSeq(result);
              setMode('PLAYBACK');
          } else {
              setPredSeq(result);
          }
      } catch(err) {
          alert("Error parsing NPY: " + err);
      }
  };

  return (
    <>
      <div className="ui-panel">
        <h1>{mode === 'SIM' ? 'Data Gen' : 'Playback'}</h1>
        
        {mode === 'SIM' && (
          <>
            <p>Points: {NUM_POINTS} | Rec Int: {RECORD_INTERVAL}s</p>
            <div className="stat-row"><span>Time:</span><span className="stat-val">{timeRef.current.toFixed(2)}s</span></div>
            <button onClick={() => setIsPlaying(!isPlaying)}>{isPlaying ? "Pause" : "Resume"}</button>
            <div className="checkbox-row"><input type="checkbox" checked={showGraph} onChange={e => setShowGraph(e.target.checked)} id="chkGraph" /><label htmlFor="chkGraph">Visualize k-NN</label></div>
            <div className="checkbox-row"><input type="checkbox" checked={showColors} onChange={e => setShowColors(e.target.checked)} id="chkColors" /><label htmlFor="chkColors">Show Colors</label></div>
            <hr style={{ borderColor: '#333', margin: '15px 0' }} />
            <div className="stat-row"><span>Rec Frames:</span><span className="stat-val">{recordedData.times.length}</span></div>
            
            {!isRecording ? (<button className="primary" onClick={startRecording}>Start Recording</button>) : (<button className="primary" style={{background: '#cc3300', borderColor: '#aa2200'}} onClick={stopAndSave}>Stop & Save</button>)}
          </>
        )}

        {mode === 'PLAYBACK' && (
            <>
                <h3 style={{fontSize:'12px', color:'#aaa', marginBottom:'5px'}}>PLAYBACK CONTROLS</h3>
                <div style={{marginBottom:'5px'}}>
                    <label style={{fontSize:'10px', display:'block', color:'#ccc'}}>1. Ground Truth (White):</label>
                    <input type="file" accept=".npy" onChange={e => handleFileUpload(e, 'GT')} style={{fontSize:'10px', color:'#fff', width:'100%'}} />
                </div>
                <div>
                    <label style={{fontSize:'10px', display:'block', color:'#ccc'}}>2. Prediction (Red):</label>
                    <input type="file" accept=".npy" onChange={e => handleFileUpload(e, 'PRED')} style={{fontSize:'10px', color:'#fff', width:'100%'}} />
                </div>
                
                <hr style={{ borderColor: '#555', margin: '15px 0' }} />
                
                <button 
                    className="primary" 
                    onClick={() => setRestartTrigger(prev => prev + 1)}
                    style={{marginBottom: '10px'}}
                >
                    RESTART ANIMATION
                </button>
                
                <div className="stat-row"><span>Total Frames:</span><span className="stat-val">{gtSeq?.shape[0] || 0}</span></div>
                <button onClick={() => { setMode('SIM'); setGtSeq(null); setPredSeq(null); }}>Back to Sim</button>
            </>
        )}
        
        {/* Loader nascosto se siamo in SIM, mostrato in PLAYBACK se necessario */}
        {mode === 'SIM' && (
            <>
                <hr style={{ borderColor: '#555', margin: '15px 0' }} />
                <button onClick={() => setMode('PLAYBACK')} style={{fontSize: '10px', opacity: 0.7}}>Switch to Playback Mode</button>
            </>
        )}
      </div>

      <Canvas style={{ background: "#111" }}>
        <PerspectiveCamera makeDefault position={[2, 2, 2]} />
        <OrbitControls target={[0, 0.5, 0]} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <gridHelper args={[10, 10, 0x444444, 0x222222]} />
        <axesHelper args={[1]} />
        
        {mode === 'SIM' ? (
             <RobotSimulation timeRef={timeRef} isPlaying={isPlaying} showGraph={showGraph} showColors={showColors} onUpdate={handleUpdate} resetTime={resetTimeRef} />
        ) : (
             <PlaybackViz groundTruthSeq={gtSeq} predictionSeq={predSeq} manualRestartTrigger={restartTrigger} />
        )}
      </Canvas>
    </>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);