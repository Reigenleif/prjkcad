import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import {
  RotateCcw,
  Box,
  Layers,
  Scissors,
  Check,
  X,
  Minus,
  Square,
  Circle as CircleIcon,
  Spline,
  Undo2,
  Trash2,
  MousePointer,
  Compass,
  ArrowRight,
  Maximize2,
} from 'lucide-react';
import {
  findClosedLoops,
  inferFacesAndLoops,
  inferCoorFromSurface,
  distance,
  sampleArc,
} from '../utils/sketchInference';

function parseBinarySTL(buffer) {
  const reader = new DataView(buffer);
  const faces = reader.getUint32(80, true);
  const vertices = new Float32Array(faces * 3 * 3);
  const normals = new Float32Array(faces * 3 * 3);

  let offset = 84;
  for (let i = 0; i < faces; i++) {
    const nx = reader.getFloat32(offset, true);
    const ny = reader.getFloat32(offset + 4, true);
    const nz = reader.getFloat32(offset + 8, true);
    offset += 12;

    for (let j = 0; j < 3; j++) {
      const vx = reader.getFloat32(offset, true);
      const vy = reader.getFloat32(offset + 4, true);
      const vz = reader.getFloat32(offset + 8, true);
      offset += 12;

      const vIdx = (i * 3 + j) * 3;
      vertices[vIdx] = vx;
      vertices[vIdx + 1] = vy;
      vertices[vIdx + 2] = vz;

      normals[vIdx] = nx;
      normals[vIdx + 1] = ny;
      normals[vIdx + 2] = nz;
    }
    offset += 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.computeBoundingBox();
  return geometry;
}

export default function CADViewport({
  stlUrl,
  viewMode = 'shaded_edges',
  showGrid = true,
  properties,
  cameraTrigger,
  // Surface Selection Props
  selectedSurface: externalSelectedSurface,
  onSelectSurface,
  // Sketch Mode Props
  inSketchMode = false,
  sketchPlane = null,
  sketchTool = 'line',
  onSetSketchTool,
  sketchEntities = [],
  onUpdateSketchEntities,
  onStartSketchOnSurface,
  onOpenExtrudeModal,
  onExitSketch,
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const meshGroupRef = useRef(null);
  const currentMeshRef = useRef(null);
  const gridHelperRef = useRef(null);
  const sketchPlaneHelperRef = useRef(null);

  // 3D Navigation Refs
  const isDraggingRef = useRef(false);
  const isPanningRef = useRef(false);
  const prevMousePosRef = useRef({ x: 0, y: 0 });
  const mouseDownPosRef = useRef({ x: 0, y: 0 });
  const cameraDistanceRef = useRef(3.5);
  const targetRef = useRef(new THREE.Vector3(0, 0, 0));

  // Surface Selection State
  const [internalSelectedSurface, setInternalSelectedSurface] = useState(null);
  const selectedSurface = externalSelectedSurface !== undefined ? externalSelectedSurface : internalSelectedSurface;
  const updateSelectedSurface = (surf) => {
    setInternalSelectedSurface(surf);
    onSelectSurface?.(surf);
  };
  const [showTangentOptions, setShowTangentOptions] = useState(false);
  const [loadedModelName, setLoadedModelName] = useState(null);

  // 2D Sketch Drawing Canvas State
  const svgOverlayRef = useRef(null);
  const [zoomScale, setZoomScale] = useState(160); // px per CAD unit
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 }); // px from center
  const [snapToGrid, setSnapToGrid] = useState(true);
  const [mouseCoord, setMouseCoord] = useState({ u: 0, v: 0 });
  const [drawingState, setDrawingState] = useState(null); // { mode, startPt, intermediatePt, ... }

  // Initialize Three.js Scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#f4f6f8');
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 500);
    camera.position.set(2.5, 2.0, 2.5);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // CAD Lighting Setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    // Floor Grid
    const gridHelper = new THREE.GridHelper(4, 40, 0x94a3b8, 0xe2e8f0);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);
    gridHelperRef.current = gridHelper;

    // Coordinate Axes Tripod
    const axesHelper = new THREE.AxesHelper(0.6);
    scene.add(axesHelper);

    // Mesh Group
    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    // Sketch Plane visual helper
    const planeGeo = new THREE.PlaneGeometry(2, 2);
    const planeMat = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      wireframe: false,
    });
    const sketchPlaneMesh = new THREE.Mesh(planeGeo, planeMat);
    sketchPlaneMesh.visible = false;
    scene.add(sketchPlaneMesh);
    sketchPlaneHelperRef.current = sketchPlaneMesh;

    // Animation Loop
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Resize Observer
    const resizeObserver = new ResizeObserver(() => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
      renderer.dispose();
    };
  }, []);

  // Update Grid Visibility
  useEffect(() => {
    if (gridHelperRef.current) {
      gridHelperRef.current.visible = showGrid && !inSketchMode;
    }
  }, [showGrid, inSketchMode]);

  // Load and Render STL
  useEffect(() => {
    if (!stlUrl || !meshGroupRef.current) return;

    fetch(stlUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.arrayBuffer();
      })
      .then((buffer) => {
        const geometry = parseBinarySTL(buffer);
        geometry.computeVertexNormals();

        // Center geometry
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        const center = new THREE.Vector3();
        bbox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        // Clear existing mesh
        while (meshGroupRef.current.children.length > 0) {
          meshGroupRef.current.remove(meshGroupRef.current.children[0]);
        }

        // Material creation
        const material = new THREE.MeshStandardMaterial({
          color: 0xdde2e8,
          metalness: 0.15,
          roughness: 0.35,
          wireframe: viewMode === 'wireframe',
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        meshGroupRef.current.add(mesh);
        currentMeshRef.current = mesh;

        // Add Edges if viewMode == 'shaded_edges'
        if (viewMode === 'shaded_edges') {
          const edgesGeo = new THREE.EdgesGeometry(geometry, 28);
          const edgesMat = new THREE.LineBasicMaterial({ color: 0x1e293b, linewidth: 1.5 });
          const edgesMesh = new THREE.LineSegments(edgesGeo, edgesMat);
          meshGroupRef.current.add(edgesMesh);
        }

        // Auto-frame camera based on bounding box
        const size = new THREE.Vector3();
        bbox.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z, 0.5);
        cameraDistanceRef.current = maxDim * 2.5;

        if (!inSketchMode) {
          resetCameraView('iso');
        }

        setLoadedModelName(stlUrl.split('/').pop());
      })
      .catch((err) => {
        console.error('Failed to load STL in CAD viewport:', err);
      });
  }, [stlUrl, viewMode]);

  // Align camera orthogonal to sketch plane when entering sketch mode
  useEffect(() => {
    if (inSketchMode && sketchPlane && cameraRef.current) {
      const camera = cameraRef.current;
      const normal = sketchPlane.normal || [0, 0, 1];
      const origin = sketchPlane.coor
        ? [sketchPlane.coor.coor_tx, sketchPlane.coor.coor_ty, sketchPlane.coor.coor_tz]
        : [0, 0, 0];

      const d = 2.5;
      targetRef.current.set(origin[0], origin[1], origin[2]);

      // Position camera along normal
      camera.position.set(
        origin[0] + normal[0] * d,
        origin[1] + normal[1] * d,
        origin[2] + normal[2] * d
      );

      // Camera up vector
      if (Math.abs(normal[1]) > 0.9) {
        camera.up.set(0, 0, normal[1] > 0 ? -1 : 1);
      } else {
        camera.up.set(0, 1, 0);
      }

      camera.lookAt(targetRef.current);

      // Show sketch plane helper
      if (sketchPlaneHelperRef.current) {
        sketchPlaneHelperRef.current.visible = true;
        sketchPlaneHelperRef.current.position.set(origin[0], origin[1], origin[2]);
        sketchPlaneHelperRef.current.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          new THREE.Vector3(normal[0], normal[1], normal[2]).normalize()
        );
      }
    } else {
      if (sketchPlaneHelperRef.current) {
        sketchPlaneHelperRef.current.visible = false;
      }
      if (cameraRef.current) {
        cameraRef.current.up.set(0, 1, 0);
      }
    }
  }, [inSketchMode, sketchPlane]);

  // Camera preset handler
  const resetCameraView = (type) => {
    const camera = cameraRef.current;
    if (!camera) return;

    if (targetRef.current) {
      targetRef.current.set(0, 0, 0);
    }

    const d = cameraDistanceRef.current || 3.0;
    camera.up.set(0, 1, 0);

    if (type === 'top') {
      camera.position.set(0, d, 0.001);
      camera.up.set(0, 0, -1);
    } else if (type === 'front') {
      camera.position.set(0, 0, d);
    } else if (type === 'right') {
      camera.position.set(d, 0, 0);
    } else {
      // Isometric
      camera.position.set(d * 0.7, d * 0.6, d * 0.7);
    }
    camera.lookAt(targetRef.current || new THREE.Vector3(0, 0, 0));
  };

  useEffect(() => {
    if (cameraTrigger) {
      resetCameraView(cameraTrigger);
    }
  }, [cameraTrigger]);

  // Raycasting for Face / Surface Selection
  const handleMeshSurfaceClick = (event) => {
    if (inSketchMode || !currentMeshRef.current || !cameraRef.current || !rendererRef.current) return;

    const rect = rendererRef.current.domElement.getBoundingClientRect();
    const mouseX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const mouseY = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), cameraRef.current);

    const intersects = raycaster.intersectObject(currentMeshRef.current, false);

    if (intersects && intersects.length > 0) {
      const hit = intersects[0];
      if (hit.face) {
        // Transform normal to world space
        const normalMatrix = new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld);
        const worldNormal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
        const worldPoint = hit.point.clone();

        const nx = Math.abs(worldNormal.x);
        const ny = Math.abs(worldNormal.y);
        const nz = Math.abs(worldNormal.z);
        const isFlat = Math.abs(nx - 1) < 0.05 || Math.abs(ny - 1) < 0.05 || Math.abs(nz - 1) < 0.05;

        const inferredCoor = inferCoorFromSurface(worldPoint, worldNormal);

        updateSelectedSurface({
          point: worldPoint,
          normal: worldNormal,
          coor: inferredCoor,
          isFlat,
          screenPos: { x: event.clientX - rect.left, y: event.clientY - rect.top },
        });
        setShowTangentOptions(false);
      }
    } else {
      // Clicked on empty space: clear surface selection
      updateSelectedSurface(null);
      setShowTangentOptions(false);
    }
  };

  // 3D Viewport Mouse Interactions
  const handleMouseDown = (e) => {
    if (inSketchMode) return;

    if (e.button === 0) {
      isDraggingRef.current = true;
    } else if (e.button === 1 || e.button === 2) {
      isPanningRef.current = true;
    }
    prevMousePosRef.current = { x: e.clientX, y: e.clientY };
    mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (inSketchMode) return;

    const deltaX = e.clientX - prevMousePosRef.current.x;
    const deltaY = e.clientY - prevMousePosRef.current.y;
    prevMousePosRef.current = { x: e.clientX, y: e.clientY };

    const camera = cameraRef.current;
    if (!camera) return;

    if (isDraggingRef.current) {
      // Orbit around target: sliding/dragging right rotates view naturally
      const rotSpeed = 0.006;
      const target = targetRef.current;
      const dx = camera.position.x - target.x;
      const dz = camera.position.z - target.z;
      const radius = Math.sqrt(dx * dx + dz * dz);
      let angle = Math.atan2(dz, dx);

      angle += deltaX * rotSpeed;
      camera.position.x = target.x + radius * Math.cos(angle);
      camera.position.z = target.z + radius * Math.sin(angle);
      camera.position.y += deltaY * rotSpeed * (cameraDistanceRef.current * 0.5);
      camera.lookAt(target);
    } else if (isPanningRef.current) {
      // Pan camera along camera plane
      const panSpeed = 0.0015 * cameraDistanceRef.current;
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      camera.matrix.extractBasis(right, up, new THREE.Vector3());

      const move = new THREE.Vector3()
        .addScaledVector(right, deltaX * panSpeed)
        .addScaledVector(up, deltaY * panSpeed);

      camera.position.add(move);
      targetRef.current.add(move);
    }
  };

  const handleMouseUp = (e) => {
    if (inSketchMode) return;

    const distMoved = Math.hypot(
      e.clientX - mouseDownPosRef.current.x,
      e.clientY - mouseDownPosRef.current.y
    );

    // Only trigger mesh click if user clicked directly on the 3D canvas (not on a button/badge/modal)
    if (distMoved < 4 && e.button === 0 && e.target === rendererRef.current?.domElement) {
      handleMeshSurfaceClick(e);
    }

    isDraggingRef.current = false;
    isPanningRef.current = false;
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const camera = cameraRef.current;
    if (!camera) return;

    if (inSketchMode) {
      // Zoom 2D Sketch Overlay
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      setZoomScale((prev) => Math.min(600, Math.max(40, prev * factor)));
      return;
    }

    // Support horizontal scroll (trackpad 2-finger swipe / wheel tilt)
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY) && Math.abs(e.deltaX) > 1.5) {
      const rotSpeed = 0.003;
      const target = targetRef.current;
      const dx = camera.position.x - target.x;
      const dz = camera.position.z - target.z;
      const radius = Math.sqrt(dx * dx + dz * dz);
      let angle = Math.atan2(dz, dx);

      angle -= e.deltaX * rotSpeed;
      camera.position.x = target.x + radius * Math.cos(angle);
      camera.position.z = target.z + radius * Math.sin(angle);
      camera.lookAt(target);
      return;
    }

    const zoomSpeed = 0.001;
    const factor = 1 + e.deltaY * zoomSpeed;
    camera.position.multiplyScalar(factor);
    cameraDistanceRef.current = camera.position.length();
  };

  // Convert SVG screen coordinates to sketch plane CAD units (u, v)
  const screenToCad = (clientX, clientY) => {
    const svg = svgOverlayRef.current;
    if (!svg) return { u: 0, v: 0 };
    const rect = svg.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;

    const centerX = rect.width / 2 + panOffset.x;
    const centerY = rect.height / 2 + panOffset.y;

    let u = (px - centerX) / zoomScale;
    let v = -(py - centerY) / zoomScale; // standard CAD coordinate: +v is up

    if (snapToGrid) {
      const step = 0.05;
      u = Math.round(u / step) * step;
      v = Math.round(v / step) * step;
    }

    return { u: Number(u.toFixed(3)), v: Number(v.toFixed(3)) };
  };

  // Convert sketch plane CAD units (u, v) to SVG screen coordinates (px, py)
  const cadToScreen = (u, v) => {
    const svg = svgOverlayRef.current;
    const width = svg?.clientWidth || 800;
    const height = svg?.clientHeight || 600;

    const centerX = width / 2 + panOffset.x;
    const centerY = height / 2 + panOffset.y;

    const px = centerX + u * zoomScale;
    const py = centerY - v * zoomScale;
    return { x: px, y: py };
  };

  // Live Closed Loops Analysis
  const loopsAnalysis = useMemo(() => {
    if (!inSketchMode || sketchEntities.length === 0) return null;
    return inferFacesAndLoops(sketchEntities, 0.06);
  }, [inSketchMode, sketchEntities]);

  // Sketch Drawing Event Handlers
  const handleSketchPointerMove = (e) => {
    const cad = screenToCad(e.clientX, e.clientY);
    setMouseCoord(cad);

    if (drawingState) {
      setDrawingState((prev) => ({
        ...prev,
        currentPt: [cad.u, cad.v],
      }));
    }
  };

  const handleSketchPointerDown = (e) => {
    if (e.button !== 0) return; // Left click only
    const cad = screenToCad(e.clientX, e.clientY);
    const pt = [cad.u, cad.v];

    if (sketchTool === 'line') {
      if (!drawingState || drawingState.mode !== 'line') {
        // Start line chain
        setDrawingState({
          mode: 'line',
          startPt: pt,
          currentPt: pt,
          chainStart: pt,
        });
      } else {
        // Place line segment
        const from = drawingState.startPt;
        const to = pt;

        // Check if close to chainStart (loop closure)
        const isNearStart =
          drawingState.chainStart &&
          distance(to, drawingState.chainStart) < 0.08 &&
          sketchEntities.length > 1;

        const finalTo = isNearStart ? drawingState.chainStart : to;

        if (distance(from, finalTo) > 0.01) {
          const newEntity = {
            type: 'line',
            sx: from[0],
            sy: from[1],
            ex: finalTo[0],
            ey: finalTo[1],
          };
          const updated = [...sketchEntities, newEntity];
          onUpdateSketchEntities?.(updated);

          if (isNearStart) {
            // Closed loop! Finish chain
            setDrawingState(null);
          } else {
            // Continue chain
            setDrawingState({
              mode: 'line',
              startPt: finalTo,
              currentPt: finalTo,
              chainStart: drawingState.chainStart,
            });
          }
        }
      }
    } else if (sketchTool === 'rectangle') {
      if (!drawingState || drawingState.mode !== 'rectangle') {
        setDrawingState({
          mode: 'rectangle',
          corner1: pt,
          currentPt: pt,
        });
      } else {
        // Commit 4 rectangle lines
        const c1 = drawingState.corner1;
        const c2 = pt;
        const minX = Math.min(c1[0], c2[0]);
        const maxX = Math.max(c1[0], c2[0]);
        const minY = Math.min(c1[1], c2[1]);
        const maxY = Math.max(c1[1], c2[1]);

        if (maxX - minX > 0.02 && maxY - minY > 0.02) {
          const rectLines = [
            { type: 'line', sx: minX, sy: minY, ex: maxX, ey: minY },
            { type: 'line', sx: maxX, sy: minY, ex: maxX, ey: maxY },
            { type: 'line', sx: maxX, sy: maxY, ex: minX, ey: maxY },
            { type: 'line', sx: minX, sy: maxY, ex: minX, ey: minY },
          ];
          onUpdateSketchEntities?.([...sketchEntities, ...rectLines]);
        }
        setDrawingState(null);
      }
    } else if (sketchTool === 'circle') {
      if (!drawingState || drawingState.mode !== 'circle') {
        setDrawingState({
          mode: 'circle',
          center: pt,
          currentPt: pt,
        });
      } else {
        // Commit circle
        const c = drawingState.center;
        const r = distance(c, pt);
        if (r > 0.02) {
          const circleEnt = {
            type: 'circle',
            cx: c[0],
            cy: c[1],
            r: Number(r.toFixed(3)),
          };
          onUpdateSketchEntities?.([...sketchEntities, circleEnt]);
        }
        setDrawingState(null);
      }
    } else if (sketchTool === 'arc') {
      if (!drawingState || drawingState.mode !== 'arc_step1') {
        // Step 1: Start point
        setDrawingState({
          mode: 'arc_step1',
          startPt: pt,
          currentPt: pt,
        });
      } else if (drawingState.mode === 'arc_step1') {
        // Step 2: Mid point
        setDrawingState({
          mode: 'arc_step2',
          startPt: drawingState.startPt,
          midPt: pt,
          currentPt: pt,
        });
      } else if (drawingState.mode === 'arc_step2') {
        // Step 3: End point
        const s = drawingState.startPt;
        const m = drawingState.midPt;
        const ePt = pt;
        if (distance(s, ePt) > 0.02) {
          const arcEnt = {
            type: 'arc',
            sx: s[0],
            sy: s[1],
            mx: m[0],
            my: m[1],
            ex: ePt[0],
            ey: ePt[1],
          };
          onUpdateSketchEntities?.([...sketchEntities, arcEnt]);
        }
        setDrawingState(null);
      }
    }
  };

  const handleSketchKeyDown = (e) => {
    if (e.key === 'Escape') {
      setDrawingState(null);
    } else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
      handleUndo();
    }
  };

  const handleUndo = () => {
    if (sketchEntities.length > 0) {
      onUpdateSketchEntities?.(sketchEntities.slice(0, -1));
    }
  };

  const handleClear = () => {
    onUpdateSketchEntities?.([]);
    setDrawingState(null);
  };

  return (
    <div
      className="center-viewport"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onContextMenu={(e) => {
        e.preventDefault();
        setDrawingState(null);
      }}
      onWheel={handleWheel}
      tabIndex={0}
      onKeyDown={handleSketchKeyDown}
    >
      {/* 3D Three.js Canvas Container */}
      <div ref={containerRef} className="three-canvas-container" />

      {/* Surface Selection Floating Helper Badge */}
      {selectedSurface && !inSketchMode && (
        <div
          className="surface-selection-badge"
          onMouseDown={(e) => e.stopPropagation()}
          onMouseUp={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Compass size={16} color="#3b82f6" />
            <span>
              <b>{selectedSurface.isFlat ? 'Flat Surface' : 'Curved Surface'} Selected</b> (Normal: [{selectedSurface.normal.x.toFixed(2)},{' '}
              {selectedSurface.normal.y.toFixed(2)}, {selectedSurface.normal.z.toFixed(2)}])
            </span>
          </div>

          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <button
              className="surface-action-btn"
              onClick={(e) => {
                e.stopPropagation();
                onStartSketchOnSurface?.(selectedSurface.coor, selectedSurface.normal);
                updateSelectedSurface(null);
              }}
            >
              <span>+ New Sketch on Face</span>
            </button>

            {/* Tangent Ref only shown when selected surface is NOT flat */}
            {!selectedSurface.isFlat && (
              <button
                className="surface-action-btn secondary"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowTangentOptions(!showTangentOptions);
                }}
                title="For curved faces: align tangent to surface"
              >
                <span>Tangent Ref...</span>
              </button>
            )}

            <button
              className="header-icon-btn"
              onClick={(e) => {
                e.stopPropagation();
                updateSelectedSurface(null);
              }}
              style={{ color: '#94a3b8', padding: 2 }}
            >
              <X size={14} />
            </button>
          </div>

          {/* Tangent Reference Options Dropdown (only for curved surfaces) */}
          {!selectedSurface.isFlat && showTangentOptions && (
            <div
              style={{
                position: 'absolute',
                bottom: 42,
                right: 0,
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: 6,
                padding: 6,
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
                zIndex: 40,
                width: 200,
                boxShadow: 'var(--shadow-lg)',
              }}
              onMouseDown={(e) => e.stopPropagation()}
              onMouseUp={(e) => e.stopPropagation()}
            >
              <div style={{ fontSize: 10, color: '#94a3b8', padding: '2px 6px', fontWeight: 600 }}>
                ALIGN TANGENT TO REFERENCE:
              </div>
              <button
                className="nav-item-btn"
                style={{ textAlign: 'left', padding: '4px 8px', fontSize: 11 }}
                onClick={(e) => {
                  e.stopPropagation();
                  const coor = {
                    ...selectedSurface.coor,
                    coor_euax: 0.0,
                    coor_euay: 0.0,
                    coor_euaz: 0.0,
                  };
                  onStartSketchOnSurface?.(coor, [0, 0, 1]);
                  updateSelectedSurface(null);
                }}
              >
                Front Plane (XY) Colinear
              </button>
              <button
                className="nav-item-btn"
                style={{ textAlign: 'left', padding: '4px 8px', fontSize: 11 }}
                onClick={(e) => {
                  e.stopPropagation();
                  const coor = {
                    ...selectedSurface.coor,
                    coor_euax: -90.0,
                    coor_euay: 0.0,
                    coor_euaz: 0.0,
                  };
                  onStartSketchOnSurface?.(coor, [0, 1, 0]);
                  updateSelectedSurface(null);
                }}
              >
                Top Plane (XZ) Colinear
              </button>
              <button
                className="nav-item-btn"
                style={{ textAlign: 'left', padding: '4px 8px', fontSize: 11 }}
                onClick={(e) => {
                  e.stopPropagation();
                  const coor = {
                    ...selectedSurface.coor,
                    coor_euax: 0.0,
                    coor_euay: 90.0,
                    coor_euaz: 0.0,
                  };
                  onStartSketchOnSurface?.(coor, [1, 0, 0]);
                  updateSelectedSurface(null);
                }}
              >
                Right Plane (YZ) Colinear
              </button>
            </div>
          )}
        </div>
      )}

      {/* 2D Interactive Sketch Mode Overlay */}
      {inSketchMode && (
        <div
          className="sketch-mode-container"
          onPointerMove={handleSketchPointerMove}
          onPointerDown={handleSketchPointerDown}
        >
          {/* Sketch Floating Toolbar */}
          <div className="sketch-toolbar">
            <span style={{ fontSize: 11.5, fontWeight: 700, color: '#2563eb', padding: '0 6px' }}>
              SKETCH [{sketchPlane?.name || 'Plane'}]
            </span>

            <div style={{ height: 16, width: 1, background: '#cbd5e1' }} />

            <button
              className={`sketch-tool-btn ${sketchTool === 'line' ? 'active' : ''}`}
              onClick={() => onSetSketchTool?.('line')}
              title="Line tool (Click to chain segments, close loop)"
            >
              <Minus size={15} />
              <span>Line</span>
            </button>

            <button
              className={`sketch-tool-btn ${sketchTool === 'rectangle' ? 'active' : ''}`}
              onClick={() => onSetSketchTool?.('rectangle')}
              title="Rectangle tool (Click two opposite corners)"
            >
              <Square size={15} />
              <span>Rectangle</span>
            </button>

            <button
              className={`sketch-tool-btn ${sketchTool === 'circle' ? 'active' : ''}`}
              onClick={() => onSetSketchTool?.('circle')}
              title="Circle tool (Click center, drag radius)"
            >
              <CircleIcon size={15} />
              <span>Circle</span>
            </button>

            <button
              className={`sketch-tool-btn ${sketchTool === 'arc' ? 'active' : ''}`}
              onClick={() => onSetSketchTool?.('arc')}
              title="3-Point Arc (Start, Mid, End)"
            >
              <Spline size={15} />
              <span>3-Pt Arc</span>
            </button>

            <div style={{ height: 16, width: 1, background: '#cbd5e1' }} />

            <button
              className={`sketch-tool-btn ${snapToGrid ? 'active' : ''}`}
              onClick={() => setSnapToGrid(!snapToGrid)}
              title="Snap to 0.05 unit grid"
            >
              <span>Snap: {snapToGrid ? 'On' : 'Off'}</span>
            </button>

            <button className="sketch-tool-btn" onClick={handleUndo} title="Undo last drawn segment (Ctrl+Z)">
              <Undo2 size={14} />
            </button>

            <button className="sketch-tool-btn" onClick={handleClear} title="Clear all sketch entities">
              <Trash2 size={14} color="#ef4444" />
            </button>

            <div style={{ height: 16, width: 1, background: '#cbd5e1' }} />

            {/* Extrude Button */}
            <button
              className="sketch-tool-btn"
              onClick={onOpenExtrudeModal}
              disabled={!loopsAnalysis?.success}
              style={{
                background: loopsAnalysis?.success ? '#16a34a' : '#e2e8f0',
                color: loopsAnalysis?.success ? '#ffffff' : '#94a3b8',
                fontWeight: 600,
                cursor: loopsAnalysis?.success ? 'pointer' : 'not-allowed',
              }}
              title={loopsAnalysis?.success ? 'Extrude this profile' : 'Draw a closed profile to extrude'}
            >
              <Check size={14} />
              <span>Extrude Feature</span>
            </button>

            <button className="sketch-tool-btn" onClick={onExitSketch} title="Exit sketch without applying">
              <X size={14} color="#ef4444" />
            </button>
          </div>

          {/* SVG Canvas for Drawing and Real-time Rendering */}
          <svg
            ref={svgOverlayRef}
            style={{ width: '100%', height: '100%', cursor: 'crosshair', userSelect: 'none' }}
          >
            {/* Grid Lines */}
            <defs>
              <pattern
                id="sketch-grid-minor"
                width={zoomScale * 0.1}
                height={zoomScale * 0.1}
                patternUnits="userSpaceOnUse"
              >
                <path
                  d={`M ${zoomScale * 0.1} 0 L 0 0 0 ${zoomScale * 0.1}`}
                  fill="none"
                  stroke="#e2e8f0"
                  strokeWidth="0.8"
                />
              </pattern>
              <pattern
                id="sketch-grid-major"
                width={zoomScale * 0.5}
                height={zoomScale * 0.5}
                patternUnits="userSpaceOnUse"
              >
                <rect width={zoomScale * 0.5} height={zoomScale * 0.5} fill="url(#sketch-grid-minor)" />
                <path
                  d={`M ${zoomScale * 0.5} 0 L 0 0 0 ${zoomScale * 0.5}`}
                  fill="none"
                  stroke="#cbd5e1"
                  strokeWidth="1.2"
                />
              </pattern>
            </defs>

            <rect width="100%" height="100%" fill="url(#sketch-grid-major)" />

            {/* Sketch Origin Axes */}
            {(() => {
              const origin = cadToScreen(0, 0);
              return (
                <g>
                  {/* U Axis (Horizontal Red) */}
                  <line
                    x1={0}
                    y1={origin.y}
                    x2="100%"
                    y2={origin.y}
                    stroke="rgba(239, 68, 68, 0.4)"
                    strokeWidth="1.5"
                    strokeDasharray="4 4"
                  />
                  {/* V Axis (Vertical Green) */}
                  <line
                    x1={origin.x}
                    y1={0}
                    x2={origin.x}
                    y2="100%"
                    stroke="rgba(34, 197, 94, 0.4)"
                    strokeWidth="1.5"
                    strokeDasharray="4 4"
                  />
                  {/* Origin Circle */}
                  <circle cx={origin.x} cy={origin.y} r={4} fill="#2563eb" />
                </g>
              );
            })()}

            {/* Filled Regions for Detected Closed Loops */}
            {loopsAnalysis?.success &&
              loopsAnalysis.faces.map((face, fIdx) => (
                <g key={`face-${fIdx}`}>
                  {/* Outer Loop Polygon Fill */}
                  {face.outer.type === 'circle' ? (
                    <circle
                      cx={cadToScreen(face.outer.curves[0].params.circle_cx, 0).x}
                      cy={cadToScreen(0, face.outer.curves[0].params.circle_cy).y}
                      r={face.outer.curves[0].params.circle_r * zoomScale}
                      fill="rgba(37, 99, 235, 0.18)"
                      stroke="#2563eb"
                      strokeWidth="2"
                    />
                  ) : (
                    face.outer.polygon && (
                      <polygon
                        points={face.outer.polygon
                          .map((p) => {
                            const sc = cadToScreen(p[0], p[1]);
                            return `${sc.x},${sc.y}`;
                          })
                          .join(' ')}
                        fill="rgba(37, 99, 235, 0.18)"
                        stroke="#2563eb"
                        strokeWidth="2"
                      />
                    )
                  )}

                  {/* Inner Hole Cutouts Fill */}
                  {face.inners.map((inner, inIdx) =>
                    inner.type === 'circle' ? (
                      <circle
                        key={`inner-${fIdx}-${inIdx}`}
                        cx={cadToScreen(inner.curves[0].params.circle_cx, 0).x}
                        cy={cadToScreen(0, inner.curves[0].params.circle_cy).y}
                        r={inner.curves[0].params.circle_r * zoomScale}
                        fill="#f8fafc"
                        stroke="#dc2626"
                        strokeWidth="2"
                        strokeDasharray="4 3"
                      />
                    ) : (
                      inner.polygon && (
                        <polygon
                          key={`inner-${fIdx}-${inIdx}`}
                          points={inner.polygon
                            .map((p) => {
                              const sc = cadToScreen(p[0], p[1]);
                              return `${sc.x},${sc.y}`;
                            })
                            .join(' ')}
                          fill="#f8fafc"
                          stroke="#dc2626"
                          strokeWidth="2"
                          strokeDasharray="4 3"
                        />
                      )
                    )
                  )}
                </g>
              ))}

            {/* Committed Entities Rendering */}
            {sketchEntities.map((ent, idx) => {
              if (ent.type === 'line') {
                const s = cadToScreen(ent.sx, ent.sy);
                const e = cadToScreen(ent.ex, ent.ey);
                return (
                  <g key={`ent-${idx}`}>
                    <line x1={s.x} y1={s.y} x2={e.x} y2={e.y} stroke="#1e293b" strokeWidth="2.5" />
                    <circle cx={s.x} cy={s.y} r={3} fill="#0f172a" />
                    <circle cx={e.x} cy={e.y} r={3} fill="#0f172a" />
                  </g>
                );
              } else if (ent.type === 'circle') {
                const c = cadToScreen(ent.cx, ent.cy);
                const rPx = ent.r * zoomScale;
                return (
                  <g key={`ent-${idx}`}>
                    <circle cx={c.x} cy={c.y} r={rPx} fill="none" stroke="#1e293b" strokeWidth="2.5" />
                    <circle cx={c.x} cy={c.y} r={3} fill="#2563eb" />
                  </g>
                );
              } else if (ent.type === 'arc') {
                const pts = sampleArc(ent.sx, ent.sy, ent.mx, ent.my, ent.ex, ent.ey, 12);
                const dPath = pts.reduce((acc, p, i) => {
                  const sc = cadToScreen(p[0], p[1]);
                  return i === 0 ? `M ${sc.x} ${sc.y}` : `${acc} L ${sc.x} ${sc.y}`;
                }, '');
                return <path key={`ent-${idx}`} d={dPath} fill="none" stroke="#1e293b" strokeWidth="2.5" />;
              }
              return null;
            })}

            {/* Dynamic Rubber-Band Previews */}
            {drawingState && drawingState.currentPt && (
              <g>
                {drawingState.mode === 'line' && (
                  (() => {
                    const s = cadToScreen(drawingState.startPt[0], drawingState.startPt[1]);
                    const cur = cadToScreen(drawingState.currentPt[0], drawingState.currentPt[1]);
                    return (
                      <>
                        <line
                          x1={s.x}
                          y1={s.y}
                          x2={cur.x}
                          y2={cur.y}
                          stroke="#2563eb"
                          strokeWidth="2"
                          strokeDasharray="4 3"
                        />
                        <circle cx={cur.x} cy={cur.y} r={4} fill="#2563eb" />
                      </>
                    );
                  })()
                )}

                {drawingState.mode === 'rectangle' && (
                  (() => {
                    const c1 = cadToScreen(drawingState.corner1[0], drawingState.corner1[1]);
                    const c2 = cadToScreen(drawingState.currentPt[0], drawingState.currentPt[1]);
                    const minX = Math.min(c1.x, c2.x);
                    const minY = Math.min(c1.y, c2.y);
                    const width = Math.abs(c2.x - c1.x);
                    const height = Math.abs(c2.y - c1.y);
                    return (
                      <rect
                        x={minX}
                        y={minY}
                        width={width}
                        height={height}
                        fill="rgba(37, 99, 235, 0.1)"
                        stroke="#2563eb"
                        strokeWidth="2"
                        strokeDasharray="4 3"
                      />
                    );
                  })()
                )}

                {drawingState.mode === 'circle' && (
                  (() => {
                    const c = cadToScreen(drawingState.center[0], drawingState.center[1]);
                    const rCad = distance(drawingState.center, drawingState.currentPt);
                    const rPx = rCad * zoomScale;
                    return (
                      <>
                        <circle
                          cx={c.x}
                          cy={c.y}
                          r={rPx}
                          fill="rgba(37, 99, 235, 0.1)"
                          stroke="#2563eb"
                          strokeWidth="2"
                          strokeDasharray="4 3"
                        />
                        <circle cx={c.x} cy={c.y} r={3} fill="#2563eb" />
                      </>
                    );
                  })()
                )}
              </g>
            )}
          </svg>

          {/* Sketch Bottom Status Bar */}
          <div className="sketch-status-bar">
            <div>
              <b>Coords: </b>
              <span style={{ fontFamily: 'var(--font-mono)' }}>
                U: {mouseCoord.u.toFixed(2)}, V: {mouseCoord.v.toFixed(2)}
              </span>
            </div>

            <div style={{ height: 14, width: 1, background: '#cbd5e1' }} />

            <div>
              {loopsAnalysis?.success ? (
                <span style={{ color: '#16a34a', fontWeight: 600 }}>
                  ✓ {loopsAnalysis.facesCount} Face ({loopsAnalysis.loopsCount} closed loops) detected
                </span>
              ) : (
                <span style={{ color: '#64748b' }}>
                  {drawingState
                    ? 'Click to place next point (click loop origin to close)'
                    : 'Draw closed profiles (lines/circles/arcs) to enable solid extrusion'}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Top HUD Readout (When in normal 3D mode) */}
      {!inSketchMode && (
        <div className="viewport-hud-top">
          <div className="hud-pill">
            <Box size={13} color="#2563eb" />
            <span>{loadedModelName || 'Empty Model'}</span>
          </div>

          {properties && properties.volume > 0 && (
            <div className="hud-pill">
              <span>Vol: <b>{properties.volume.toFixed(3)}</b></span>
            </div>
          )}

          {properties?.bbox?.max && (
            <div className="hud-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5 }}>
              <span>
                Size: [
                {(properties.bbox.max[0] - properties.bbox.min[0]).toFixed(2)} ×{' '}
                {(properties.bbox.max[1] - properties.bbox.min[1]).toFixed(2)} ×{' '}
                {(properties.bbox.max[2] - properties.bbox.min[2]).toFixed(2)}]
              </span>
            </div>
          )}
        </div>
      )}

      {/* Quick Camera Buttons */}
      {!inSketchMode && (
        <div className="viewport-hud-right">
          <button className="hud-btn" onClick={() => resetCameraView('iso')} title="Reset Camera (Iso)">
            <RotateCcw size={15} />
          </button>
          <button className="hud-btn" onClick={() => resetCameraView('top')} title="Top View">
            <span style={{ fontSize: 10, fontWeight: 700 }}>T</span>
          </button>
          <button className="hud-btn" onClick={() => resetCameraView('front')} title="Front View">
            <span style={{ fontSize: 10, fontWeight: 700 }}>F</span>
          </button>
          <button className="hud-btn" onClick={() => resetCameraView('right')} title="Right View">
            <span style={{ fontSize: 10, fontWeight: 700 }}>R</span>
          </button>
        </div>
      )}

      {/* Viewport View Cube */}
      {!inSketchMode && (
        <div className="view-cube-widget">
          <button className="cube-btn" onClick={() => resetCameraView('top')}>TOP</button>
          <button className="cube-btn" onClick={() => resetCameraView('front')}>FRT</button>
          <button className="cube-btn" onClick={() => resetCameraView('right')}>RGT</button>
          <button className="cube-btn" onClick={() => resetCameraView('iso')}>ISO</button>
          <div style={{ background: '#e2e8f0', borderRadius: 2 }} />
          <button className="cube-btn" onClick={() => resetCameraView('front')}>BOT</button>
        </div>
      )}
    </div>
  );
}
