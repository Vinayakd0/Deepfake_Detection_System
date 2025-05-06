import React, { useState } from "react";
import axios from "axios";
import "./Detect.css"; // Keep CSS separate
import DetectImage from "../../Assets/facesearch.png";
import { Line } from "react-chartjs-2";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

const Detect = () => {
  const [file, setFile] = useState(null);
  const [videoURL, setVideoURL] = useState(null);
  const [output, setOutput] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false); // New Loading State

  const onFileSelect = (event) => {
    if (event.target.files && event.target.files[0]) {
      const videoFile = event.target.files[0];
      setFile(videoFile);
      setVideoURL(URL.createObjectURL(videoFile)); // Generate preview URL
    }
  };

  const uploadFile2Backend = async () => {
    if (!file) {
      alert("Please select a video file first.");
      return;
    }

    setLoading(true); // Start loading

    const formData = new FormData();
    formData.append("video", file);

    console.log("Uploading file:", file.name);

    try {
      const res = await axios.post("http://127.0.0.1:3500/Detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Full API Response:", res.data);

      setOutput(res.data.output || "Unknown");
      setConfidence(res.data.confidence || 0);
      setGraphData(res.data.graph || { frames: [], probabilities: [] });
      setError(null);
    } catch (error) {
      console.error("Upload error:", error);
      setError("Error uploading video. Please try again.");
    } finally {
      setLoading(false); // Stop loading when request is done
    }
  };

  return (
    <div className="background">
      <h1 className="detect-heading">IS YOUR VIDEO FAKE? CHECK IT!</h1>

      {/* Upload Section */}
      <div className="upload-section">
        <label htmlFor="video-upload" className="button">+ ADD VIDEO</label>
        <input id="video-upload" type="file" accept="video/*" onChange={onFileSelect} hidden />
        {file && (
          <button
            id="submitBtn"
            className="submit-button"
            onClick={uploadFile2Backend}
            disabled={loading} // Disable button while loading
          >
            {loading ? "Processing..." : "Submit"}
          </button>
        )}
      </div>

      {/* Video Preview */}
      {videoURL && (
        <div className="video-preview">
          <h2>Uploaded Video Preview:</h2>
          <video controls width="400">
            <source src={videoURL} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      )}

      {/* Show Loading Indicator when uploading */}
      {loading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Processing video, please wait...</p>
        </div>
      )}

      {/* Displaying Deepfake Detection Image */}
      <img src={DetectImage} alt="Deepfake Detection" className="detect-image" />

      {/* Error Handling */}
      {error && <p className="error-message">{error}</p>}

      {/* Results Section */}
      {output && !loading && (
        <div className="result-container">
          <h2 className="result">Result: <span>{output}</span></h2>
          <h2 className="result">Confidence: <span>{confidence.toFixed(2)}%</span></h2>

          {/* Confidence Progress Bar */}
          <div className="progress-bar-container">
            <div
              className="progress-bar"
              style={{
                width: {confidence},
                backgroundColor: confidence > 75 ? "green" : confidence > 50 ? "orange" : "red",
              }}
            >
              {confidence.toFixed(2)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Detect;