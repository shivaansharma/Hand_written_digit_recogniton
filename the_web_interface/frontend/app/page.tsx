// pages/index.tsx
'use client';

import DrawingCanvas from "@/components/Canvas";
import ModelParameters from "@/components/io";

const Home = () => {
 

  return (
    <div className="h-full w-full">
    
      <div></div><DrawingCanvas />
     
      <ModelParameters  />
    </div>
  );
};

export default Home;