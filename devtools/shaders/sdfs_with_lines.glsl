#version 330 core

out vec4 fragColor;

uniform vec2 _iResolution;
uniform uint _maxSteps;
uniform vec3 _camPose;
uniform vec3 _lookingAt;
uniform vec3 _camUp;
uniform float _fx;
uniform float _stopEpsilon;
uniform float _tmax;

float map(in vec3 p)
{
    float s1 = length(p - vec3(-0.2, 0.0, 0.0)) - 0.5;
    float s2 = length(p - vec3(0.2, 0.0, 0.0)) - 0.5;
    return min(s1, s2);
}

float map_lines(in vec3 p, in float epsilon)
{
    float s1 = length(p - vec3(-0.2, 0.0, 0.0)) - 0.5;
    float s2 = length(p - vec3(0.2, 0.0, 0.0)) - 0.5;
    return float(abs(s1) <= epsilon && abs(s2) <= epsilon);
}

vec3 calcNormal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0);
    const float eps = 0.0005;
    return normalize( e.xyy*map( pos + e.xyy*eps ) + 
					  e.yyx*map( pos + e.yyx*eps ) + 
					  e.yxy*map( pos + e.yxy*eps ) + 
					  e.xxx*map( pos + e.xxx*eps ) );
}

vec4 mainImage( in vec2 fragCoord )
{
    vec3 cam_norm = normalize(_lookingAt - _camPose);
    vec3 cam_right = normalize(cross(cam_norm, _camUp));
    vec3 cam_down = normalize(cross(cam_right, cam_norm));
    float fy = (_fx / _iResolution.x) * _iResolution.y;

    vec2 normalized_coord = (fragCoord / _iResolution) - 0.5;
    normalized_coord.x = normalized_coord.x * _fx;
    normalized_coord.y = normalized_coord.y * fy;

    vec3 p0 = cam_right * normalized_coord.x + cam_down * normalized_coord.y;
    p0 += _camPose;

        
    vec3 tot = vec3(0.0);

    // raymarch
    float t = 0.0;
    float h = 0.0;
    vec3 pos;
    for( uint i=0u; i<_maxSteps; i++ )
    {
        pos = p0 + t*cam_norm;
        h = map(pos);
        if( h<_stopEpsilon || t>_tmax ) break;
        t += h;
    }
    vec4 color = vec4(0.0);
    if( h<_stopEpsilon ) {
        color.w = 1.0;
        vec3 nor = calcNormal(pos);
        float dif = clamp( dot(nor,vec3(0.57703)), 0.0, 1.0 ) * 0.2;
        float amb = 1.4 + 0.3*dot(nor,vec3(0.0,1.0,0.0));
        color.xyz = vec3(0.2,0.3,0.4)*amb + vec3(0.8,0.7,0.5)*dif;
        color.xyz *= (1.0 - map_lines(pos, _stopEpsilon * 80.0 * _fx));
    }

    return color;
}

void main() {
    fragColor = mainImage(gl_FragCoord.xy);
}
