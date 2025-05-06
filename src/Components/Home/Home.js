import React from "react";
import "./Home.css";

const Home = () => {
  return (
    <div className="content">
      <h1 className="heading">
        <span className="highlight">DEEP</span> FAKE
      </h1>
      <p className="para">
        Deepfake is a synthetic media technique that uses artificial intelligence (AI), 
        specifically deep learning, to manipulate or generate visual and audio content 
        that appears real. Deepfakes often involve altering videos, images, or voices 
        to create realistic but false representations of people, making it seem like 
        they said or did something they never actually did.
      </p>
    </div>
  );
};

export default Home;
