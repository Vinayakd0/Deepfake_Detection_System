import React, { useState } from "react";
import axios from "axios";

const VideoUpload = () => {
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (event) => {
    setVideo(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!video) {
      alert("Please select a video.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("video", video);

    try {
      const response = await axios.post("http://127.0.0.1:3500/Detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data);
    } catch (error) {
      console.error("Error uploading video:", error);
      alert("Failed to upload video.");
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h2>Deepfake Detection</h2>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Processing..." : "Upload & Detect"}
      </button>

      {result && (
        <div className="result">
          <h3>Detection Result:</h3>
          <p><strong>Output:</strong> {result.output}</p>
          <p><strong>Confidence:</strong> {result.confidence.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
