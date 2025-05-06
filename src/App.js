import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./Components/Home/Home";
import Detect from "./Components/Detect/Detect";
import Navbar from "./Components/Navbar/Navbar";
import VideoUpload from "./Components/VideoUpload";
import BackgroundImage from "./Assets/bgimage.jpg";
import "./App.css";

function App() {
  return (
    <React.Fragment>
      <div className="App" style={{ backgroundImage: `url(${BackgroundImage})`, backgroundSize: "cover" }}>
        <Router>
          <Navbar />
          <Routes>
            <Route exact path='/' element={<Home />} />
            <Route exact path='/Detect' element={<Detect />} />
            <Route exact path='/upload' element={<VideoUpload />} />
          </Routes>
        </Router>
      </div>
    </React.Fragment>
  );
}

export default App;
