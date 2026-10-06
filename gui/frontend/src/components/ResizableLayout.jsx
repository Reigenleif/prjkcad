import React, { useState, useRef, useEffect } from 'react';

export default function ResizableLayout({
  partManagerComponent,
  chatDivisionComponent,
  centerViewportComponent,
  bottomEditorComponent,
  showBottomEditor,
}) {
  const [sidebarWidth, setSidebarWidth] = useState(380);
  const [topHeightPercent, setTopHeightPercent] = useState(45);

  const isResizingWidthRef = useRef(false);
  const isResizingHeightRef = useRef(false);

  const sidebarRef = useRef(null);

  // Width Resizing (Left sidebar overall width)
  const handleWidthMouseDown = (e) => {
    e.preventDefault();
    isResizingWidthRef.current = true;
    document.body.style.cursor = 'col-resize';
  };

  // Height Resizing (Between Part Manager & Chat)
  const handleHeightMouseDown = (e) => {
    e.preventDefault();
    isResizingHeightRef.current = true;
    document.body.style.cursor = 'row-resize';
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isResizingWidthRef.current) {
        const newWidth = Math.max(280, Math.min(680, e.clientX));
        setSidebarWidth(newWidth);
      } else if (isResizingHeightRef.current && sidebarRef.current) {
        const rect = sidebarRef.current.getBoundingClientRect();
        const offsetY = e.clientY - rect.top;
        const totalHeight = rect.height;
        const newPercent = Math.max(15, Math.min(85, (offsetY / totalHeight) * 100));
        setTopHeightPercent(newPercent);
      }
    };

    const handleMouseUp = () => {
      isResizingWidthRef.current = false;
      isResizingHeightRef.current = false;
      document.body.style.cursor = 'default';
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <div className="workspace-container">
      {/* Resizable Left Sidebar */}
      <div
        ref={sidebarRef}
        className="left-sidebar"
        style={{ width: `${sidebarWidth}px`, minWidth: `${sidebarWidth}px` }}
      >
        {/* Division 1: CAD Part Manager (SolidWorks Part Tree) */}
        <div
          className="sidebar-division"
          style={{ height: `${topHeightPercent}%`, minHeight: '120px' }}
        >
          <div className="division-header">
            <span>CAD Part Manager</span>
          </div>
          {partManagerComponent}
        </div>

        {/* Vertical Resizer Handle between Part Manager and Chat */}
        <div
          className="resizer-vertical"
          onMouseDown={handleHeightMouseDown}
          title="Drag up/down to adjust panel height"
        />

        {/* Division 2: Chat Division */}
        <div
          className="sidebar-division"
          style={{ height: `${100 - topHeightPercent}%`, minHeight: '140px' }}
        >
          <div className="division-header">
            <span>AI Assistant & DualSeq</span>
          </div>
          {chatDivisionComponent}
        </div>
      </div>

      {/* Horizontal Resizer Handle between Left Sidebar and Center Section */}
      <div
        className="resizer-horizontal"
        onMouseDown={handleWidthMouseDown}
        title="Drag left/right to adjust sidebar width"
      />

      {/* Center Section: CAD Viewport and Bottom Sequence Editor */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {centerViewportComponent}
        {showBottomEditor && bottomEditorComponent}
      </div>
    </div>
  );
}
