import styled from "styled-components";
import { NavLink as Link } from "react-router-dom";

export const NavContainer = styled.nav`
  display: flex;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  background: rgba(0, 0, 0, 0.7);
  padding: 1rem;
  z-index: 1000;
  justify-content: center;
  gap: 2rem;
`;

export const NavItem = styled(Link)`
  color: white;
  font-size: 1.2rem;
  font-weight: bold;
  text-decoration: none;
  transition: 0.3s ease-in-out;
  padding: 0.5rem 1rem;

  &.active {
    color: #16F7FA;
    text-decoration: underline;
  }

  &:hover {
    color: #16F7FA;
    transform: scale(1.1);
  }
`;
