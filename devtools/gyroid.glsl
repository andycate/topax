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

// float map(in vec3 p)
// {
//     vec3 scaled = p * 1.0;
//     float gyroid = 0.33 * abs(dot(sin(scaled.xyz), cos(scaled.yzx)));
//     gyroid = gyroid - 0.05;
//     float sphere = length(p) - 5.0;
//     return max(sphere, gyroid);
// }

const float sdfin_var0 = 1.0;
const float sdfin_var1 = 0.05;
const float sdfin_var2 = 5.0;

float map(in vec3 p)
{
    float d;
    float local_var0;
    float local_var1;
    float local_var2;
    float local_var3;
    vec3 local_var4;
    vec3 local_var5;
    local_var5 = p * sdfin_var0;
    local_var4 = sin(local_var5);
    vec3 local_var6;
    vec3 local_var7;
    local_var7 = p * sdfin_var0;
    local_var6 = local_var7.yzx;
    local_var5 = cos(local_var6);
    local_var3 = dot(local_var4, local_var5);
    local_var2 = abs(local_var3);
    local_var1 = local_var2 * 0.33;
    local_var0 = local_var1 - sdfin_var1;
    local_var2 = length(p);
    local_var1 = local_var2 - sdfin_var2;
    d = max(local_var0, local_var1);
    
    return d;
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
    }

    return color;
}

void main() {
    fragColor = mainImage(gl_FragCoord.xy);
}
