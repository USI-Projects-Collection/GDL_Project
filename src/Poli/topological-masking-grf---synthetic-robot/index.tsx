import React, { useState, useRef, useEffect, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";

// --- Configuration ---
const NUM_POINTS = 1024;
const K_NEIGHBORS = 6;
const ARM_SEGMENTS = {
  base: { height: 0.2, radius: 0.15 },
  link1: { length: 0.6, radius: 0.08 },
  link2: { length: 0.5, radius: 0.06 },
  wrist: { length: 0.1, radius: 0.07 },
  finger: { length: 0.15, radius: 0.02 }
};

// --- NPY Helper ---

function createNpyHeader(shape: number[], dtype: string): Uint8Array {
  // Magic string: \x93NUMPY
  const magic = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]);
  const version = new Uint8Array([1, 0]); // v1.0

  // Dictionary format: {'descr': '<f4', 'fortran_order': False, 'shape': (10, 3), }
  // Note: shape must end with comma if tuple
  let shapeStr = shape.join(", ");
  if (shape.length === 1) shapeStr += ","; 
  
  let headerStr = `{ 'descr': '${dtype}', 'fortran_order': False, 'shape': (${shapeStr}), }`;
  
  // Padding to align to 64 bytes
  // Length = Magic(6) + Version(2) + HeaderLen(2) + HeaderStr
  const currentLen = 10 + headerStr.length;
  const paddingNeeded = 64 - (currentLen % 64);
  headerStr += " ".repeat(paddingNeeded - 1) + "\n";
  
  const headerBytes = new TextEncoder().encode(headerStr);
  const headerLenVal = headerBytes.length;
  
  // Little endian unsigned short for header length
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
  // data.buffer might be larger than data.byteLength if it's a view, so we slice
  const body = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Blob([header, body], { type: "application/octet-stream" });
}

// --- Math & Kinematics Helpers ---

function sampleCylinder(
  radius: number, 
  height: number, 
  numPoints: number, 
  transform: THREE.Matrix4
): Float32Array {
  const points = new Float32Array(numPoints * 3);
  const vec = new THREE.Vector3();
  for (let i = 0; i < numPoints; i++) {
    const theta = Math.random() * Math.PI * 2;
    const y = (Math.random() - 0.5) * height;
    vec.set(Math.cos(theta) * radius, y, Math.sin(theta) * radius);
    vec.applyMatrix4(transform);
    points[i * 3] = vec.x;
    points[i * 3 + 1] = vec.y;
    points[i * 3 + 2] = vec.z;
  }
  return points;
}

function sampleBox(
  width: number, 
  height: number, 
  depth: number, 
  numPoints: number, 
  transform: THREE.Matrix4
): Float32Array {
  const points = new Float32Array(numPoints * 3);
  const vec = new THREE.Vector3();
  for (let i = 0; i < numPoints; i++) {
    vec.set(
      (Math.random() - 0.5) * width,
      (Math.random() - 0.5) * height,
      (Math.random() - 0.5) * depth
    );
    vec.applyMatrix4(transform);
    points[i * 3] = vec.x;
    points[i * 3 + 1] = vec.y;
    points[i * 3 + 2] = vec.z;
  }
  return points;
}

function computeKNN(positions: Float32Array, k: number): Uint32Array {
  const n = positions.length / 3;
  const indices = new Uint32Array(n * k);
  
  for (let i = 0; i < n; i++) {
    const x1 = positions[i * 3];
    const y1 = positions[i * 3 + 1];
    const z1 = positions[i * 3 + 2];
    
    // Store [distSq, index]
    const dists = new Float32Array(n);
    // Optimization: we could just store indices and sort by computed dist, 
    // but storing dists explicitly for sorting is cleaner in JS.
    const neighborIndices = new Int32Array(n); // Using Int32 for sort comp
    
    for (let j = 0; j < n; j++) {
      const dx = x1 - positions[j * 3];
      const dy = y1 - positions[j * 3 + 1];
      const dz = z1 - positions[j * 3 + 2];
      dists[j] = dx*dx + dy*dy + dz*dz;
      neighborIndices[j] = j;
    }

    // Sort indices based on distance
    // Using a temporary array to sort is O(N log N). For 512 points, 512*9 ~ 4500 ops per point -> 2M total per frame. OK.
    neighborIndices.sort((a, b) => dists[a] - dists[b]);
    
    // Take 1..k+1 (skip self at index 0)
    for (let j = 0; j < k; j++) {
      indices[i * k + j] = neighborIndices[j + 1]; 
    }
  }
  return indices;
}

// --- Components ---

const RobotSimulation = ({ 
  timeRef, 
  isPlaying, 
  showGraph,
  showColors,
  onUpdate 
}: { 
  timeRef: React.MutableRefObject<number>, 
  isPlaying: boolean, 
  showGraph: boolean,
  showColors: boolean,
  onUpdate: (pts: Float32Array, t: number) => void
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  
  const structure = useMemo(() => {
    const totalPoints = NUM_POINTS;
    const dist = [0.15, 0.35, 0.35, 0.15]; // base, link1, link2, gripper
    const counts = dist.map(p => Math.floor(p * totalPoints));
    counts[3] += totalPoints - counts.reduce((a,b) => a+b, 0);

    const baseLocal = sampleBox(0.3, 0.2, 0.3, counts[0], new THREE.Matrix4());
    const l1Local = sampleCylinder(ARM_SEGMENTS.link1.radius, ARM_SEGMENTS.link1.length, counts[1], new THREE.Matrix4());
    const l2Local = sampleCylinder(ARM_SEGMENTS.link2.radius, ARM_SEGMENTS.link2.length, counts[2], new THREE.Matrix4());
    
    const fingerPointsPerFinger = Math.floor(counts[3] / 3);
    const f1Local = sampleCylinder(ARM_SEGMENTS.finger.radius, ARM_SEGMENTS.finger.length, fingerPointsPerFinger, new THREE.Matrix4());
    const f2Local = sampleCylinder(ARM_SEGMENTS.finger.radius, ARM_SEGMENTS.finger.length, fingerPointsPerFinger, new THREE.Matrix4());
    const f3Local = sampleCylinder(ARM_SEGMENTS.finger.radius, ARM_SEGMENTS.finger.length, counts[3] - 2*fingerPointsPerFinger, new THREE.Matrix4());

    return {
      counts,
      localPoints: [baseLocal, l1Local, l2Local, f1Local, f2Local, f3Local]
    };
  }, []);

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(NUM_POINTS * 3);
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    
    const colors = new Float32Array(NUM_POINTS * 3);
    const colorMap = [
      [0.3, 0.3, 0.3], // Base
      [0.2, 0.6, 1.0], // Link1
      [1.0, 0.5, 0.2], // Link2
      [0.8, 0.8, 0.8], 
      [0.8, 0.8, 0.8], 
      [0.8, 0.8, 0.8], 
    ];
    
    let ptr = 0;
    structure.localPoints.forEach((pts, i) => {
      const c = colorMap[i];
      for (let j=0; j<pts.length/3; j++) {
        colors[ptr*3] = c[0];
        colors[ptr*3+1] = c[1];
        colors[ptr*3+2] = c[2];
        ptr++;
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
    
    timeRef.current += delta;
    const t = timeRef.current;

    // Kinematics
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
         positions[ptr*3] = v.x;
         positions[ptr*3+1] = v.y;
         positions[ptr*3+2] = v.z;
         ptr++;
       }
    };

    apply(structure.localPoints[0], mBase);
    apply(structure.localPoints[1], mLink1);
    apply(structure.localPoints[2], mLink2);
    apply(structure.localPoints[3], fingerTransforms[0]);
    apply(structure.localPoints[4], fingerTransforms[1]);
    apply(structure.localPoints[5], fingerTransforms[2]);

    geometry.attributes.position.needsUpdate = true;
    
    if (showGraph) {
        // Compute for visualization
        const knnIndices = computeKNN(positions, K_NEIGHBORS);
        const linePos = lineGeometry.attributes.position.array as Float32Array;
        let linePtr = 0;
        
        const step = 2; 
        for(let i=0; i<NUM_POINTS; i+=step) {
           const x1 = positions[i*3];
           const y1 = positions[i*3+1];
           const z1 = positions[i*3+2];
           
           for(let k=0; k<K_NEIGHBORS; k++) {
              const neighborIdx = knnIndices[i*K_NEIGHBORS + k];
              const x2 = positions[neighborIdx*3];
              const y2 = positions[neighborIdx*3+1];
              const z2 = positions[neighborIdx*3+2];
              
              linePos[linePtr++] = x1;
              linePos[linePtr++] = y1;
              linePos[linePtr++] = z1;
              linePos[linePtr++] = x2;
              linePos[linePtr++] = y2;
              linePos[linePtr++] = z2;
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
        {showColors ? (
          <pointsMaterial key="col" size={0.03} vertexColors />
        ) : (
          <pointsMaterial key="mono" size={0.03} vertexColors={false} color="#ffffff" />
        )}
      </points>
      <lineSegments ref={linesRef} geometry={lineGeometry}>
        <lineBasicMaterial 
          key="boldLines"
          color="#ff4444"
          transparent={false}
          opacity={1.0}
        />
      </lineSegments>
    </>
  );
};


// --- Main UI & App ---

const App = () => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [showGraph, setShowGraph] = useState(true);
  const [showColors, setShowColors] = useState(true);
  const [frameCount, setFrameCount] = useState(0);
  const timeRef = useRef(0);
  
  // Data recording state
  const [isRecording, setIsRecording] = useState(false);
  
  // Buffers for storing data before final save
  const [recordedData, setRecordedData] = useState<{
    points: Float32Array[],
    knn: Uint32Array[],
    times: number[]
  }>({ points: [], knn: [], times: [] });

  const handleUpdate = (positions: Float32Array, t: number) => {
    setFrameCount(Math.floor(t * 60)); 
    
    if (isRecording) {
       // Deep copy positions
       const posCopy = new Float32Array(positions);
       // Re-compute KNN exactly for the dataset (viz might skip steps, dataset must not)
       const knn = computeKNN(positions, K_NEIGHBORS);
       
       setRecordedData(prev => ({
         points: [...prev.points, posCopy],
         knn: [...prev.knn, knn],
         times: [...prev.times, t]
       }));
    }
  };

  const startRecording = () => {
    setRecordedData({ points: [], knn: [], times: [] });
    setIsRecording(true);
    timeRef.current = 0; 
  };

  const stopAndSave = async () => {
    setIsRecording(false);
    setIsPlaying(false);
    
    const numFrames = recordedData.times.length;
    if (numFrames === 0) {
      alert("No frames recorded.");
      return;
    }

    // 1. Combine Points -> (T, N, 3)
    const pointsFlat = new Float32Array(numFrames * NUM_POINTS * 3);
    for(let i=0; i<numFrames; i++) {
        pointsFlat.set(recordedData.points[i], i * NUM_POINTS * 3);
    }
    
    // 2. Combine KNN -> (T, N, K)
    const knnFlat = new Uint32Array(numFrames * NUM_POINTS * K_NEIGHBORS);
    for(let i=0; i<numFrames; i++) {
        knnFlat.set(recordedData.knn[i], i * NUM_POINTS * K_NEIGHBORS);
    }
    
    // 3. Timestamps -> (T,)
    const timesFlat = new Float32Array(recordedData.times);

    // Create Blobs
    const pointsBlob = serializeNpy(pointsFlat, [numFrames, NUM_POINTS, 3], '<f4');
    const knnBlob = serializeNpy(knnFlat, [numFrames, NUM_POINTS, K_NEIGHBORS], '<u4');
    const timeBlob = serializeNpy(timesFlat, [numFrames], '<f4');

    // Attempt File System Access API
    // Check if API is available in this environment
    if ('showDirectoryPicker' in window) {
      try {
        const dirHandle = await (window as any).showDirectoryPicker();
        if (!dirHandle) return; // cancelled

        const pointsFile = await dirHandle.getFileHandle('points.npy', { create: true });
        const pointsWritable = await pointsFile.createWritable();
        await pointsWritable.write(pointsBlob);
        await pointsWritable.close();

        const knnFile = await dirHandle.getFileHandle('knn_indices.npy', { create: true });
        const knnWritable = await knnFile.createWritable();
        await knnWritable.write(knnBlob);
        await knnWritable.close();

        const timeFile = await dirHandle.getFileHandle('timestamps.npy', { create: true });
        const timeWritable = await timeFile.createWritable();
        await timeWritable.write(timeBlob);
        await timeWritable.close();

        alert(`Saved ${numFrames} frames successfully.`);
        return;
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
           console.warn("File system access failed, falling back to download", e);
        } else {
           // User cancelled picker
           return;
        }
      }
    }

    // Fallback: Standard Download
    const download = (blob: Blob, name: string) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    download(pointsBlob, 'points.npy');
    // Slight delay to ensure browsers process multiple downloads
    setTimeout(() => download(knnBlob, 'knn_indices.npy'), 200);
    setTimeout(() => download(timeBlob, 'timestamps.npy'), 400);
    
    alert(`Saved ${numFrames} frames (via Download).`);
  };

  return (
    <>
      <div className="ui-panel">
        <h1>Topological Data Generator</h1>
        <p>
          Simulating a 2-link robot arm with a 3-finger gripper. 
          Generating {NUM_POINTS} points per frame.
        </p>
        
        <div className="stat-row">
          <span>Simulation Time:</span>
          <span className="stat-val">{timeRef.current.toFixed(2)}s</span>
        </div>
        <div className="stat-row">
          <span>Points:</span>
          <span className="stat-val">{NUM_POINTS}</span>
        </div>
        
        <button onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? "Pause Simulation" : "Resume Simulation"}
        </button>
        
        <div className="checkbox-row">
          <input 
            type="checkbox" 
            checked={showGraph} 
            onChange={e => setShowGraph(e.target.checked)} 
            id="chkGraph"
          />
          <label htmlFor="chkGraph">Visualize k-NN Graph (k={K_NEIGHBORS})</label>
        </div>
        <div className="checkbox-row">
          <input
            type="checkbox"
            checked={showColors}
            onChange={e => setShowColors(e.target.checked)}
            id="chkColors"
          />
          <label htmlFor="chkColors">Show Colors</label>
        </div>

        <hr style={{ borderColor: '#333', margin: '15px 0' }} />
        
        <div className="stat-row">
            <span>Recorded Frames:</span>
            <span className="stat-val">{recordedData.times.length}</span>
        </div>
        
        {!isRecording ? (
             <button 
                className="primary" 
                onClick={startRecording}
             >
               Start Recording (NPY)
             </button>
        ) : (
             <button className="primary" style={{background: '#cc3300', borderColor: '#aa2200'}} onClick={stopAndSave}>
               Stop & Save
             </button>
        )}
        {isRecording && <p style={{marginTop: '10px', fontSize: '10px', color: '#fc0'}}>Recording... click Stop to save.</p>}
      </div>

      <Canvas style={{ background: "#111" }}>
        <PerspectiveCamera makeDefault position={[2, 2, 2]} />
        <OrbitControls target={[0, 0.5, 0]} />
        
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        
        <gridHelper args={[10, 10, 0x444444, 0x222222]} />
        <axesHelper args={[1]} />
        
        <RobotSimulation 
          timeRef={timeRef} 
          isPlaying={isPlaying} 
          showGraph={showGraph}
          showColors={showColors}
          onUpdate={handleUpdate}
        />
      </Canvas>
    </>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);