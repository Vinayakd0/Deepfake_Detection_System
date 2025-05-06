import React from "react";
import { NavContainer, NavItem } from "./NavbarElements";

const Navbar = () => {
  return (
    <NavContainer>
      <NavItem exact to="/">HOME</NavItem>
      <NavItem exact to="/Detect">DETECT</NavItem>
    </NavContainer>
  );
};

export default Navbar;
