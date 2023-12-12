# Dealing with objects

Flow:
- Start from SVG => Convert to Shapely
- Shapely => list of vertices in CCW
- Convex partition using C++
- Save in format json.