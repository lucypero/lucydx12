



Going through the Book of shaders.


I came up with a triangle function that you feed to a ease out cubic function. Seems useful.


Run in https://thebookofshaders.com/edit.php#05/cubicpulse.frag

Visualization with a line:



```glsl

// Author: Inigo Quiles
// Title: Cubic Pulse

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

//  Function from Iñigo Quiles
//  www.iquilezles.org/www/articles/functions/functions.htm
float cubicPulse( float c, float w, float x ){
    x = abs(x - c);
    if( x>w ) return 0.0;
    x /= w;
    return 1.0 - x*x*(3.0-2.0*x);
}

float easeInOutCubic(float x) {
return x < 0.5 ? 4. * x * x * x : 1. - pow(-2. * x + 2., 3.) / 2.;
}

float triangle(float x) {
    return abs((mod(x, 2.)) - 1.);
}

float plot(vec2 st, float pct){
  return  smoothstep( pct-0.02, pct, st.y) -
          smoothstep( pct, pct+0.02, st.y);
}

void main() {
    vec2 st = gl_FragCoord.xy/u_resolution;

    float y = easeInOutCubic(triangle(u_time));
    //y = triangle(u_time);
    
    
    vec3 color = vec3(y);

    float pct = plot(st,y);
    color = (1.0-pct)*color+pct*vec3(0.0,1.0,0.0);

    gl_FragColor = vec4(color,1.0);
}

```

Visualization with 2 colors. mixing between them:


```glsl
#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform float u_time;

vec3 colorA = vec3(0.149,0.141,0.912);
vec3 colorB = vec3(1.000,0.833,0.224);

float easeInOutCubic(float x) {
return x < 0.5 ? 4. * x * x * x : 1. - pow(-2. * x + 2., 3.) / 2.;
}

float triangle(float x) {
    return abs((mod(x, 2.)) - 1.);
}

void main() {
    vec3 color = vec3(0.0);

    float pct = easeInOutCubic(triangle(u_time * 2.880));

    // Mix uses pct (a value from 0-1) to
    // mix the two colors
    color = mix(colorA, colorB, pct);

    gl_FragColor = vec4(color,1.0);
}
```