// 2D Sketch Loop and Topology Inference Engine for Text2CAD DualSeq

export function distance(p1, p2) {
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

export function pointsEqual(p1, p2, tol = 0.05) {
  return distance(p1, p2) <= tol;
}

export function sampleArc(sx, sy, mx, my, ex, ey, count = 8) {
  // Approximate arc through (sx, sy), (mx, my), (ex, ey)
  const d = 2 * (sx * (my - ey) + mx * (ey - sy) + ex * (sy - my));
  if (Math.abs(d) < 1e-6) {
    // Collinear points: return straight points
    const pts = [];
    for (let i = 0; i <= count; i++) {
      const t = i / count;
      pts.push([sx + t * (ex - sx), sy + t * (ey - sy)]);
    }
    return pts;
  }

  const p1Sq = sx * sx + sy * sy;
  const p2Sq = mx * mx + my * my;
  const p3Sq = ex * ex + ey * ey;

  const cx = (p1Sq * (my - ey) + p2Sq * (ey - sy) + p3Sq * (sy - my)) / d;
  const cy = (p1Sq * (ex - mx) + p2Sq * (sx - ex) + p3Sq * (mx - sx)) / d;
  const r = Math.sqrt((sx - cx) * (sx - cx) + (sy - cy) * (sy - cy));

  let a1 = Math.atan2(sy - cy, sx - cx);
  let a2 = Math.atan2(my - cy, mx - cx);
  let a3 = Math.atan2(ey - cy, ex - cx);

  // Normalize angles relative to a1
  let diff2 = (a2 - a1) % (2 * Math.PI);
  if (diff2 < 0) diff2 += 2 * Math.PI;

  let diff3 = (a3 - a1) % (2 * Math.PI);
  if (diff3 < 0) diff3 += 2 * Math.PI;

  const ccw = diff2 < diff3;
  let totalSpan = diff3;
  if (!ccw) {
    totalSpan = diff3 - 2 * Math.PI;
  }

  const pts = [];
  for (let i = 0; i <= count; i++) {
    const frac = i / count;
    const a = a1 + frac * totalSpan;
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return pts;
}

export function shoelaceArea(polygon) {
  let area = 0;
  const n = polygon.length;
  if (n < 3) return 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += polygon[i][0] * polygon[j][1];
    area -= polygon[j][0] * polygon[i][1];
  }
  return area / 2;
}

export function pointInPolygon(point, polygon) {
  let inside = false;
  const x = point[0];
  const y = point[1];
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];

    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

export function findClosedLoops(entities, tol = 0.05) {
  const loops = [];

  // 1. Process Circles (always self-contained closed loops)
  for (const ent of entities) {
    if (ent.type === 'circle') {
      const cx = Number(ent.cx);
      const cy = Number(ent.cy);
      const r = Number(ent.r);
      if (r <= 0) continue;

      const area = Math.PI * r * r;
      loops.push({
        type: 'circle',
        area: area,
        absArea: area,
        bbox: [cx - r, cy - r, cx + r, cy + r],
        testPoint: [cx, cy],
        containsPoint: (p) => {
          const dx = p[0] - cx;
          const dy = p[1] - cy;
          return dx * dx + dy * dy < r * r;
        },
        curves: [
          {
            cmd: 'CIRCLE',
            params: {
              circle_cx: Number(cx.toFixed(4)),
              circle_cy: Number(cy.toFixed(4)),
              circle_r: Number(r.toFixed(4)),
            },
          },
        ],
      });
    }
  }

  // 2. Process connected curve segments (lines and arcs)
  const segments = [];
  for (const ent of entities) {
    if (ent.type === 'line') {
      segments.push({
        type: 'line',
        start: [Number(ent.sx), Number(ent.sy)],
        end: [Number(ent.ex), Number(ent.ey)],
        ent,
      });
    } else if (ent.type === 'arc') {
      segments.push({
        type: 'arc',
        start: [Number(ent.sx), Number(ent.sy)],
        mid: [Number(ent.mx), Number(ent.my)],
        end: [Number(ent.ex), Number(ent.ey)],
        ent,
      });
    }
  }

  const remaining = [...segments];

  while (remaining.length > 0) {
    const chain = [remaining.shift()];
    let extended = true;

    while (extended && remaining.length > 0) {
      extended = false;
      const currentEnd = chain[chain.length - 1].end;

      for (let i = 0; i < remaining.length; i++) {
        const seg = remaining[i];
        if (pointsEqual(currentEnd, seg.start, tol)) {
          chain.push(seg);
          remaining.splice(i, 1);
          extended = true;
          break;
        } else if (pointsEqual(currentEnd, seg.end, tol)) {
          // Reverse segment
          const reversed = {
            ...seg,
            start: seg.end,
            end: seg.start,
            reversed: true,
          };
          chain.push(reversed);
          remaining.splice(i, 1);
          extended = true;
          break;
        }
      }

      // Check if chain forms a closed loop
      if (chain.length >= 2 && pointsEqual(chain[chain.length - 1].end, chain[0].start, tol)) {
        break;
      }
    }

    // Verify if chain is closed
    if (chain.length >= 2 && pointsEqual(chain[chain.length - 1].end, chain[0].start, tol)) {
      // Discretize loop into polygon vertices
      const poly = [];
      const curves = [];

      for (const seg of chain) {
        if (seg.type === 'line') {
          poly.push(seg.start);
          curves.push({
            cmd: 'LINE',
            params: {
              line_sx: Number(seg.start[0].toFixed(4)),
              line_sy: Number(seg.start[1].toFixed(4)),
              line_ex: Number(seg.end[0].toFixed(4)),
              line_ey: Number(seg.end[1].toFixed(4)),
            },
          });
        } else if (seg.type === 'arc') {
          const arcPts = sampleArc(
            seg.start[0],
            seg.start[1],
            seg.mid[0],
            seg.mid[1],
            seg.end[0],
            seg.end[1],
            8
          );
          for (let p = 0; p < arcPts.length - 1; p++) {
            poly.push(arcPts[p]);
          }
          curves.push({
            cmd: 'ARC',
            params: {
              arc_sx: Number(seg.start[0].toFixed(4)),
              arc_sy: Number(seg.start[1].toFixed(4)),
              arc_mx: Number(seg.mid[0].toFixed(4)),
              arc_my: Number(seg.mid[1].toFixed(4)),
              arc_ex: Number(seg.end[0].toFixed(4)),
              arc_ey: Number(seg.end[1].toFixed(4)),
            },
          });
        }
      }

      const signedArea = shoelaceArea(poly);
      const absArea = Math.abs(signedArea);

      if (absArea > 1e-4) {
        // Centroid test point
        let sumX = 0;
        let sumY = 0;
        for (const pt of poly) {
          sumX += pt[0];
          sumY += pt[1];
        }
        const centroid = [sumX / poly.length, sumY / poly.length];

        loops.push({
          type: 'polygon',
          polygon: poly,
          area: signedArea,
          absArea: absArea,
          testPoint: centroid,
          containsPoint: (p) => pointInPolygon(p, poly),
          curves,
        });
      }
    }
  }

  return loops;
}

export function inferFacesAndLoops(entities, tol = 0.05) {
  const loops = findClosedLoops(entities, tol);

  if (loops.length === 0) {
    return {
      success: false,
      error: 'No closed profile or loop found in the sketch. Ensure lines connect or circles are drawn.',
      faces: [],
    };
  }

  // Sort loops by area descending (outer boundary is larger than inner hole)
  loops.sort((a, b) => b.absArea - a.absArea);

  // Hierarchy identification
  const roots = [];
  const innersMap = new Map(); // rootIndex -> [innerLoops]

  for (let i = 0; i < loops.length; i++) {
    const currentLoop = loops[i];
    let parentIndex = -1;

    // Check against all larger loops that strictly contain currentLoop's testPoint
    for (let j = 0; j < i; j++) {
      const potentialParent = loops[j];
      if (potentialParent.containsPoint(currentLoop.testPoint)) {
        parentIndex = j; // smallest enclosing larger loop
      }
    }

    if (parentIndex === -1) {
      roots.push(i);
      innersMap.set(i, []);
    } else {
      // Find ultimate root for this parent
      let rootOfParent = parentIndex;
      while (!roots.includes(rootOfParent) && rootOfParent >= 0) {
        rootOfParent--;
      }
      if (innersMap.has(rootOfParent)) {
        innersMap.get(rootOfParent).push(currentLoop);
      } else {
        roots.push(i);
        innersMap.set(i, []);
      }
    }
  }

  // Form FACE definitions
  const faces = [];
  for (const rootIdx of roots) {
    const outerLoop = loops[rootIdx];
    const innerLoops = innersMap.get(rootIdx) || [];
    faces.push({
      outer: outerLoop,
      inners: innerLoops,
    });
  }

  return {
    success: true,
    loopsCount: loops.length,
    facesCount: faces.length,
    faces,
  };
}

export function buildDualSeqSubpart({
  coorParams,
  facesResult,
  extrudeCmd = 'EXTRUDE_NEW',
  extrudeParams = {},
}) {
  const tuples = [];

  // 1. COOR
  tuples.push([
    'COOR',
    {
      coor_tx: Number(Number(coorParams.coor_tx || 0).toFixed(4)),
      coor_ty: Number(Number(coorParams.coor_ty || 0).toFixed(4)),
      coor_tz: Number(Number(coorParams.coor_tz || 0).toFixed(4)),
      coor_euax: Number(Number(coorParams.coor_euax || 0).toFixed(4)),
      coor_euay: Number(Number(coorParams.coor_euay || 0).toFixed(4)),
      coor_euaz: Number(Number(coorParams.coor_euaz || 0).toFixed(4)),
    },
  ]);

  // 2. FACEs & LOOPs
  for (const face of facesResult.faces) {
    tuples.push(['FACE', {}]);

    // Outer Loop
    tuples.push(['LOOP', {}]);
    for (const curve of face.outer.curves) {
      tuples.push([curve.cmd, curve.params]);
    }

    // Inner Hole Loops
    for (const innerLoop of face.inners) {
      tuples.push(['LOOP', {}]);
      for (const curve of innerLoop.curves) {
        tuples.push([curve.cmd, curve.params]);
      }
    }
  }

  // 3. EXTRUDE
  tuples.push([extrudeCmd, extrudeParams]);

  return tuples;
}

export function inferCoorFromSurface(point, normal) {
  // Normalize normal vector
  const len = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) || 1;
  const nx = normal.x / len;
  const ny = normal.y / len;
  const nz = normal.z / len;

  let euax = 0;
  let euay = 0;
  let euaz = 0;

  const tol = 0.05;

  if (Math.abs(nz - 1) < tol) {
    // Normal +Z (XY Top plane)
    euax = 0;
    euay = 0;
    euaz = 0;
  } else if (Math.abs(nz + 1) < tol) {
    // Normal -Z (Bottom plane)
    euax = 180;
    euay = 0;
    euaz = 0;
  } else if (Math.abs(nx - 1) < tol) {
    // Normal +X (YZ Right plane)
    euax = 0;
    euay = 90;
    euaz = 0;
  } else if (Math.abs(nx + 1) < tol) {
    // Normal -X (Left plane)
    euax = 0;
    euay = -90;
    euaz = 0;
  } else if (Math.abs(ny - 1) < tol) {
    // Normal +Y (ZX Front plane)
    euax = -90;
    euay = 0;
    euaz = 0;
  } else if (Math.abs(ny + 1) < tol) {
    // Normal -Y (Back plane)
    euax = 90;
    euay = 0;
    euaz = 0;
  } else {
    // General orientation: pitch and yaw
    const pitch = Math.asin(-ny);
    const yaw = Math.atan2(nx, nz);
    euax = (pitch * 180) / Math.PI;
    euay = (yaw * 180) / Math.PI;
    euaz = 0;
  }

  return {
    coor_tx: Number(point.x.toFixed(4)),
    coor_ty: Number(point.y.toFixed(4)),
    coor_tz: Number(point.z.toFixed(4)),
    coor_euax: Number(euax.toFixed(4)),
    coor_euay: Number(euay.toFixed(4)),
    coor_euaz: Number(euaz.toFixed(4)),
  };
}
