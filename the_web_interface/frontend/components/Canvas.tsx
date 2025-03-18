'use client';

import { useRef, useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

function throttle(func: (...args: any[]) => void, limit: number): (...args: any[]) => void {
  let inThrottle = false;
  return function (...args: any[]) {
    if (!inThrottle) {
      func.apply(null, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

const DrawingCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [ctx, setCtx] = useState<CanvasRenderingContext2D | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [responseData, setResponseData] = useState<any>(null);
  const [preprocessedImage, setPreprocessedImage] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      if (context) {
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.lineWidth = 12;
        context.strokeStyle = '#000000';
        context.fillStyle = 'white';
        context.fillRect(0, 0, canvas.width, canvas.height);
        setCtx(context);
      }
    }
  }, []);

  const startDrawing = (e: React.MouseEvent) => {
    if (!canvasRef.current || !ctx) return;
    const { offsetX, offsetY } = e.nativeEvent;
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = (e: React.MouseEvent) => {
    if (!isDrawing || !ctx) return;
    const { offsetX, offsetY } = e.nativeEvent;
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    throttledSendImage();
  };

  const clearCanvas = () => {
    if (!canvasRef.current || !ctx) return;
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setResponseData(null);
    setPreprocessedImage(null);
  };

  const getProcessedImage = (): string | null => {
    if (!canvasRef.current) return null;
    const canvas = canvasRef.current;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');

    if (!tempCtx) return null;

    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;

    for (let i = 0; i < pixels.length; i += 4) {
      const grayscaleValue = pixels[i];
      const invertedValue = 255 - grayscaleValue;
      pixels[i] = pixels[i + 1] = pixels[i + 2] = invertedValue;
    }

    tempCtx.putImageData(imageData, 0, 0);
    return tempCanvas.toDataURL('image/png');
  };

  const sendImage = async () => {
    const processedImage = getProcessedImage();
    if (!processedImage) return;

    setIsProcessing(true);

    try {
      const response = await fetch('http://127.0.0.1:5002/api/process-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layers: 3,
          nodes: [64, 32, 10],
          epochs: 10,
          learning_rate: 0.001,
          image: processedImage,
        }),
      });

      const result = await response.json();
      setResponseData(result);

      if (result.visualization) {
        setPreprocessedImage(`data:image/png;base64,${result.visualization}`);
      }

      console.log('Server response:', result);
    } catch (error) {
      console.error('Error sending image:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const throttledSendImage = throttle(sendImage, 300);

  return (
    <div className="max-w-lg mx-auto p-6 rounded-lg shadow-xl bg-white h-full w-full">
      <h2 className="text-3xl font-semibold text-center text-gray-800 mb-4">Draw a Digit</h2>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        className="border border-gray-300 rounded-lg bg-gray-100 cursor-crosshair"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      />
      <div className="flex justify-between mt-4">
        <button className="w-1/2 p-2 bg-blue-500 text-white rounded-lg" onClick={clearCanvas}>Clear</button>
        <button className="w-1/2 p-2 ml-2 bg-green-500 text-white rounded-lg" onClick={sendImage} disabled={isProcessing}>
          {isProcessing ? 'Processing...' : 'Send'}
        </button>
      </div>

      {responseData && (
        <div className="mt-4 text-center">
          <h3 className="text-lg font-semibold text-black">Prediction: {responseData.prediction}</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={responseData.probabilities.map((p: number, i: number) => ({ index: i, value: p }))}>
              <XAxis dataKey="index" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="value" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
          {preprocessedImage && (
            <div className="mt-4 h-full w-full">
              <h4 className="text-md font-semibold text-gray-700">Preprocessed Image</h4>
              <img
                src={preprocessedImage}
                alt="Preprocessed Digit"
                className="mt-2 border border-gray-300 rounded-lg"
                style={{ width: '112px', height: '112px' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DrawingCanvas;