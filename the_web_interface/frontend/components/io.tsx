'use client';

import { useState } from 'react';

const TrainModel: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  
  const handleTrainModel = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5002/api/train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      const result = await response.json();
      console.log("Train Model Response:", result);

      if (response.ok) {
        alert("Model trained successfully!");
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error training model:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-6 rounded-lg shadow-xl bg-white text-center m-10">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">Train Model</h2>
      <button
        onClick={handleTrainModel}
        className="w-full p-3 bg-green-500 text-white font-semibold rounded-lg hover:bg-green-600 transition"
        disabled={loading}
      >
        {loading ? "Training..." : "Train Model"}
      </button>
    </div>
  );
};

export default TrainModel;