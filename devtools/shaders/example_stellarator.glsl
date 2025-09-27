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
uniform uint _mode;

const float PI = 3.1415926536;
const float PI2 = PI * 2.0;

const vec3 sdf_colors[1] = vec3[1](vec3(0.3, 0.6, 0.8));

const vec4 axis_R_1to4 = vec4(1.97056327e-01, 2.31540618e-02, 2.54675518e-03, 2.49185652e-04);
const vec4 axis_R_5to8 = vec4(2.07685824e-05, 1.37175832e-06, 4.57863499e-08, -4.56675923e-09);
const vec4 axis_Z_1to4 = vec4(-1.52100411e-01, -2.07089289e-02, -2.47290475e-03, -2.55552655e-04);
const vec4 axis_Z_5to8 = vec4(-2.21394096e-05, -1.47903248e-06, -5.28109155e-08, 1.73691498e-08);

const vec4 theta_start_coef_nsin_1to4 = vec4(-0.42863913, -0.18206658, -0.08002005, -0.03721455);

const mat4 theta_coef_msin_1to4_ncos_1to4 = mat4(
    -0.020532860962712333, -0.21961060069307833, -0.04657785118867989, -0.0565575298107702,
    -0.09105364201075132, -0.2550709532095046, 0.12716704294012104, 0.0212458344643971,
    -0.03080111295257002, -0.1694976652390685, 0.14306589132280198, 0.0529432234667337,
    -0.01452958032683308, -0.10154566829012546, 0.12495414471431544, 0.049650299064026474
);
const mat4 theta_coef_msin_1to4_ncos_5to8 = mat4(
    -0.010408573208893442, -0.06539241186682035, 0.09450584550136232, 0.04260731331865716,
    -0.008916701428857601, -0.043165116086959186, 0.06700334799055507, 0.033064462211286505,
    -0.010928321572326165, -0.029774900218276334, 0.04579762417615643, 0.023892368434796946,
    -0.011414941953273451, -0.020640429850218252, 0.026091901277807444, 0.014855451134551967
);
const mat4 theta_coef_mcos_1to4_nsin_1to4 = mat4(
    0.2789075182521256, 0.49063150951400625, 0.031241735600437602, 0.10735773338340066,
    0.057937522701778876, 0.2617821919599211, -0.08959478124127246, -0.002216267481353151,
    0.03911136594720294, 0.1590861114559701, -0.14600294180437098, -0.04866989409586371,
    0.020507480195005298, 0.10536365896133977, -0.12630068342661943, -0.053914903849172933
);
const mat4 theta_coef_mcos_1to4_nsin_5to8 = mat4(
    0.012791388160219614, 0.06706091807907458, -0.09400257903819917, -0.04262041389714202,
    0.010755594397433916, 0.04478867229897964, -0.06652509771830462, -0.032942623486484905,
    0.010427936132103964, 0.031663441077384176, -0.044576669314793094, -0.02374472344620071,
    0.008805592846374537, 0.02214052160576823, -0.02502028287365168, -0.014409932884261956
);
const vec4 theta_coef_msin_1to4_n0 = vec4(
    -0.10405430781622936, 0.1721689498990879, -0.1512412635641782, -0.0388977503862564
);
const vec4 theta_coef_m0_nsin_1to4 = vec4(
    -0.7838996542244232, -0.2040812388840856, -0.07912473188107762, -0.03985903939017489
);
const vec4 theta_coef_m0_nsin_5to8 = vec4(
    -0.0225055881755856, -0.015450866181378708, -0.01291431395566557, -0.010595392603088056
);

// -------- R COEFFICIENTS ---------
const mat4 R_coef_msin_1to4_nsin_1to4 = mat4(
    -0.16910825533170573, 0.012376238490717189, 0.0007393814448036585, 0.0010845481230021647,
    -0.026915287123527448, 0.0035175912683862117, -0.0008191473874614593, 0.00018685233683148097,
    -0.00279690333620624, 0.0015305634244714896, 0.0002268361063537393, -0.00019328461310439462,
    -8.513478902183095e-06, 6.988678887939531e-05, -9.612756343086964e-05, 5.078229720895943e-05
);

const mat4 R_coef_msin_5to8_nsin_1to4 = mat4(
    -0.00018436349688728948, -2.2751075238350795e-06, -8.89183717586583e-06, -9.857031536719895e-08, 
    6.754807129442091e-05, 1.726179341900177e-05, 2.494357148781524e-06, 3.851620160661886e-07, 
    -2.3823638830237647e-05, 1.236972667777244e-06, 7.997515740021219e-07, 1.5722023913115076e-07, 
    6.366207978054329e-06, 2.1005514472124503e-06, -1.4784981042261254e-06, 2.5961318341835087e-08
);

const mat4 R_coef_msin_1to4_nsin_5to8 = mat4(
    2.3075638505119087e-06, 2.269040444870203e-06, -1.8378788279239543e-06, 4.7185983444582856e-07, 
    1.2478342859484978e-07, -1.0849247496106808e-07, 7.040225916467776e-08, 2.319561441255777e-07, 
    -2.3557716984704852e-08, -5.122074956937111e-09, -4.026292395259625e-08, -1.029608483692375e-09, 
    2.091540852315281e-09, -3.347918940984995e-10, 7.62538566700838e-10, 5.984480200212767e-10
);

const mat4 R_coef_msin_5to8_nsin_5to8 = mat4(
    2.514409008276808e-06, -9.844154247450484e-07, -1.619378717884307e-07, 2.7190231956315924e-09, 
    -1.0569691922412212e-07, -4.509577861678662e-08, 4.129830119648322e-07, -7.674048618064794e-09, 
    1.0629105235627623e-07, 1.3965642784785618e-08, 3.137488883590532e-07, -9.691652817538962e-09, 
    1.1578145663302794e-11, -1.881981674093158e-09, -4.682805575853149e-10, -2.0740599232993023e-09
);

const mat4 R_coef_mcos_1to4_ncos_1to4 = mat4(
    -0.07444245337161019, 0.02232487298948669, 0.002654153845159638, 0.0018099827088001329, 
    -0.008180260335814798, 0.005667614043965372, -0.001310019729300885, 0.00026246508942486435, 
    -0.0009639287349862136, 0.0022462712740724785, 0.00012946218809717998, -0.00028908322176002927, 
    -1.204559803060089e-06, 0.00013306553299195014, -0.00010092806229270365, 2.1078328674691427e-05
);

const mat4 R_coef_mcos_5to8_ncos_1to4 = mat4(
    -0.00013066482426796888, -8.43617929031662e-06, -3.894147280977162e-06, 5.57546080853332e-09, 
    6.437343164610398e-05, 1.3362198180979265e-05, 5.360627776581816e-06, -7.240802666070179e-09, 
    -4.801991472278834e-05, 4.833939607428563e-07, 1.863047199844688e-06, 3.5167375499826417e-08, 
    -1.8894516641984622e-06, 3.027065942048931e-06, -8.861135540146302e-07, -1.893535152298721e-08
);

const mat4 R_coef_mcos_1to4_ncos_5to8 = mat4(
    1.9394707069267045e-06, -4.6732577614019313e-07, -7.751470105417916e-06, -2.8756755886546724e-06, 
    1.2500042679758463e-08, 5.419170959581834e-08, -1.3733232148516348e-07, -5.664092112457603e-08, 
    -8.048785283624081e-09, -6.156372332114357e-09, 4.5794531033474085e-09, 1.1511045064614298e-08, 
    1.3374046375556663e-10, 1.1230113184202667e-09, 5.037244052031434e-10, -1.0003620683977943e-09
);

const mat4 R_coef_mcos_5to8_ncos_5to8 = mat4(
    1.8078139798988261e-06, -5.098278475493114e-07, -1.782667349559798e-07, -1.1892398158869851e-08, 
    -5.204737826812862e-07, 1.1509711298518136e-07, 2.0684836419723866e-07, -4.726062646241953e-10, 
    -2.8207981320589435e-08, 1.2207419946441717e-07, 3.2936709831972804e-07, -5.755509751069342e-09, 
    -3.117046905721077e-09, 1.0527951544110183e-08, -2.91851776098286e-09, -9.497376943060344e-09
);

const vec4 R_coef_mcos_1to4_n0 = vec4(0.1465740269477129, 0.03217751415396675, 0.0033725230757324846, -0.0003953841471958331);
const vec4 R_coef_mcos_5to8_n0 = vec4(-9.341637123605795e-06, -6.7194456006839145e-06, -5.717445414140186e-06, -3.2652568895800336e-08);
const vec4 R_coef_m0_ncos_1to4 = vec4(0.17781990105761777, 0.01028650836856859, -0.0006259966371603055, -7.397844723879479e-05);
const vec4 R_coef_m0_ncos_5to8 = vec4(-7.345839566398615e-07, -5.6886962661444005e-08, -1.4604858357527167e-09, 4.857244735252631e-10);
const float R_coef_m0_n0 = 1.0;

// -------- Z COEFFICIENTS ---------
const mat4 Z_coef_mcos_1to4_nsin_1to4 = mat4(
    0.03739911249506528, 0.00014992326185244077, 0.0006012986764830388, -0.0006090808664253625, 
    0.010196086530775205, -0.00772425101259351, 0.0006593925643240597, 6.206660786641748e-05, 
    0.0009219289935576533, -0.0017496933598855258, 0.00016003027009505282, 6.461236901045141e-05, 
    1.252728373030915e-05, -0.00019229484498045537, 9.428568521363197e-05, 2.4824673821301685e-05
);

const mat4 Z_coef_mcos_5to8_nsin_1to4 = mat4(
    -0.00010811629514243924, 1.154863693639481e-05, -6.508550617192279e-06, 1.1943659918211136e-07, 
    4.115419708967074e-05, -1.2679384744115936e-05, -3.341107178015402e-06, 5.828718969770345e-08, 
    2.371007604958224e-05, 1.1650372647166195e-05, -3.8112112993247055e-06, 6.095940064196328e-08, 
    8.573628241226233e-07, -2.2257185123088794e-06, 1.0226538699837418e-06, -3.3771733677417006e-08
);

const mat4 Z_coef_mcos_1to4_nsin_5to8 = mat4(
    -4.864245640414028e-07, -7.943693912876469e-07, 3.433686121940363e-06, 2.6374159700767438e-06, 
    3.672658301320866e-08, 4.157853853480165e-08, 1.2211953181087438e-07, 4.547627172176217e-07, 
    -9.893832369016017e-09, -1.4533690434273349e-08, -2.5460894680416095e-09, -7.0258286664801165e-09, 
    6.628711691467791e-10, 7.189487548805496e-10, -1.474438778398352e-09, 3.66547873338739e-11
);

const mat4 Z_coef_mcos_5to8_nsin_5to8 = mat4(
    3.930658791907887e-06, -6.555382146210448e-07, -5.739244829591997e-07, 3.9790458937037055e-08, 
    3.9383442946510547e-07, -1.979538008746542e-08, -6.819199162234991e-07, 4.318048019758911e-09, 
    -9.963275373632314e-09, -6.43482282799407e-08, -2.609911387160831e-07, 1.9935616432352686e-08, 
    -5.510800049688302e-10, -5.607998297690501e-10, -1.7738675974986452e-09, -6.093496820776005e-10
);

const mat4 Z_coef_msin_1to4_ncos_1to4 = mat4(
    -0.13725153137988633, -0.0036873847064226774, -0.0015599114851288524, 2.186321481261081e-05, 
    -0.02715149050337673, 0.006262993194342203, -0.00010467592953829628, 7.566029641731409e-05, 
    -0.002543182768046054, 0.001082273246326757, -7.129397540598594e-05, 3.482718089654828e-05, 
    -4.710356585224622e-05, 0.00013488548826397833, -7.367007103305599e-05, -6.73961140842395e-08
);

const mat4 Z_coef_msin_5to8_ncos_1to4 = mat4(
    6.813440607579283e-05, -4.339200106617598e-06, 1.2517445293404895e-06, -7.014051254936206e-08, 
    -2.6112646956849153e-05, 2.0442021688524764e-05, 2.7398091257051035e-07, 2.2386845261320226e-08, 
    -1.2813524175763119e-05, -8.644038368443856e-06, 1.2873402873547714e-06, 8.479966624132389e-08, 
    4.509149167303188e-06, 1.1656778988743358e-06, -1.7729863343093536e-06, 1.589092282928492e-07
);

const mat4 Z_coef_msin_1to4_ncos_5to8 = mat4(
    1.6784230976895644e-06, 9.798902277752667e-08, 1.3845063187746913e-07, 1.2828049854203433e-06, 
    -4.734135361946155e-09, -2.0544941354724177e-08, -2.2382103863176497e-08, -6.232871972767322e-08, 
    -1.5730749378702045e-08, -9.822592889197321e-09, -4.667401273409194e-08, -2.6744379304418694e-08, 
    -1.216595664251304e-10, -1.4321664061776593e-10, 2.2788857315202995e-10, -2.8470145597215554e-09
);

const mat4 Z_coef_msin_5to8_ncos_5to8 = mat4(
    -2.8523654146242337e-06, -2.9280596270877113e-07, 9.593182290245176e-07, -6.564074575124934e-09, 
    1.264618477046913e-08, -2.2220557613148243e-07, 8.145760750303128e-07, 1.8740882139209264e-08, 
    3.602846762498973e-08, 2.8716960346378433e-08, 1.891913800572474e-07, -1.2235866006867376e-08, 
    4.663079658058367e-10, -1.593457709554052e-10, 2.3835304866669368e-09, 1.840151545666879e-09
);

const vec4 Z_coef_msin_1to4_n0 = vec4(-0.24313441064699526, -0.026604330123313983, -0.007592243529224662, -0.0002549573114899858);
const vec4 Z_coef_msin_5to8_n0 = vec4(-7.08967072281184e-05, 1.1170811318312778e-06, 5.398010810217722e-06, 5.569126614002186e-07);
const vec4 Z_coef_m0_nsin_1to4 = vec4(-0.11953841059171537, -0.00875887958835589, 0.00043665508838077995, 0.00010057994122673418);
const vec4 Z_coef_m0_nsin_5to8 = vec4(6.013807919962239e-07, -1.3683319532788955e-09, -1.6869223918688115e-08, 7.657206740036527e-10);

float map_sdf0(in vec3 p)
{
    float zeta = atan(p.y, p.x) * 2.0;
    vec2 r0 = vec2(1.01964961e+00, 0.0);
    vec4 zeta_cos_1to4 = cos(vec4(zeta, 2.0 * zeta, 3.0 * zeta, 4.0 * zeta));
    vec4 zeta_cos_5to8 = cos(vec4(5.0 * zeta, 6.0 * zeta, 7.0 * zeta, 8.0 * zeta));
    vec4 zeta_sin_1to4 = sin(vec4(zeta, 2.0 * zeta, 3.0 * zeta, 4.0 * zeta));
    vec4 zeta_sin_5to8 = sin(vec4(5.0 * zeta, 6.0 * zeta, 7.0 * zeta, 8.0 * zeta));
    r0.x += dot(zeta_cos_1to4, axis_R_1to4);
    r0.x += dot(zeta_cos_5to8, axis_R_5to8);
    r0.y += dot(zeta_sin_1to4, axis_Z_1to4);
    r0.y += dot(zeta_sin_5to8, axis_Z_5to8);
    vec2 r = vec2(length(p.xy), p.z) - r0;

    float theta_initial = atan(r.y, -r.x) + PI;
    theta_initial -= dot(theta_start_coef_nsin_1to4, zeta_sin_1to4);
    // theta_initial = mod(theta_initial, PI2);

    vec4 theta_cos_1to4 = cos(vec4(theta_initial, 2.0 * theta_initial, 3.0 * theta_initial, 4.0 * theta_initial));
    vec4 theta_sin_1to4 = sin(vec4(theta_initial, 2.0 * theta_initial, 3.0 * theta_initial, 4.0 * theta_initial));

    float theta_final = dot(theta_coef_msin_1to4_ncos_1to4 * theta_sin_1to4, zeta_cos_1to4);
    theta_final +=      dot(theta_coef_msin_1to4_ncos_5to8 * theta_sin_1to4, zeta_cos_5to8);
    theta_final +=      dot(theta_coef_mcos_1to4_nsin_1to4 * theta_cos_1to4, zeta_sin_1to4);
    theta_final +=      dot(theta_coef_mcos_1to4_nsin_5to8 * theta_cos_1to4, zeta_sin_5to8);
    theta_final +=      dot(theta_coef_msin_1to4_n0, theta_sin_1to4);
    theta_final +=      dot(theta_coef_m0_nsin_1to4, zeta_sin_1to4);
    theta_final +=      dot(theta_coef_m0_nsin_5to8, zeta_sin_5to8);

    vec4 theta_final_cos_1to4 = cos(vec4(theta_final, 2.0 * theta_final, 3.0 * theta_final, 4.0 * theta_final));
    vec4 theta_final_cos_5to8 = cos(vec4(5.0 * theta_final, 6.0 * theta_final, 7.0 * theta_final, 8.0 * theta_final));
    vec4 theta_final_sin_1to4 = sin(vec4(theta_final, 2.0 * theta_final, 3.0 * theta_final, 4.0 * theta_final));
    vec4 theta_final_sin_5to8 = sin(vec4(5.0 * theta_final, 6.0 * theta_final, 7.0 * theta_final, 8.0 * theta_final));

    // now compute closest point (add 1.0 to R at end because of precision issues)
    float R = dot(R_coef_msin_1to4_nsin_1to4 * theta_final_sin_1to4, zeta_sin_1to4);
    R +=      dot(R_coef_msin_5to8_nsin_1to4 * theta_final_sin_5to8, zeta_sin_1to4);
    R +=      dot(R_coef_msin_1to4_nsin_5to8 * theta_final_sin_1to4, zeta_sin_5to8);
    R +=      dot(R_coef_msin_5to8_nsin_5to8 * theta_final_sin_5to8, zeta_sin_5to8);

    R +=      dot(R_coef_mcos_1to4_ncos_1to4 * theta_final_cos_1to4, zeta_cos_1to4);
    R +=      dot(R_coef_mcos_5to8_ncos_1to4 * theta_final_cos_5to8, zeta_cos_1to4);
    R +=      dot(R_coef_mcos_1to4_ncos_5to8 * theta_final_cos_1to4, zeta_cos_5to8);
    R +=      dot(R_coef_mcos_5to8_ncos_5to8 * theta_final_cos_5to8, zeta_cos_5to8);

    R +=      dot(R_coef_mcos_1to4_n0, theta_final_cos_1to4);
    R +=      dot(R_coef_mcos_5to8_n0, theta_final_cos_5to8);

    R +=      dot(R_coef_m0_ncos_1to4, zeta_cos_1to4);
    R +=      dot(R_coef_m0_ncos_5to8, zeta_cos_5to8);

    R +=      R_coef_m0_n0;

    // compute Z
    float Z = dot(Z_coef_mcos_1to4_nsin_1to4 * theta_final_cos_1to4, zeta_sin_1to4);
    Z +=      dot(Z_coef_mcos_5to8_nsin_1to4 * theta_final_cos_5to8, zeta_sin_1to4);
    Z +=      dot(Z_coef_mcos_1to4_nsin_5to8 * theta_final_cos_1to4, zeta_sin_5to8);
    Z +=      dot(Z_coef_mcos_5to8_nsin_5to8 * theta_final_cos_5to8, zeta_sin_5to8);

    Z +=      dot(Z_coef_msin_1to4_ncos_1to4 * theta_final_sin_1to4, zeta_cos_1to4);
    Z +=      dot(Z_coef_msin_5to8_ncos_1to4 * theta_final_sin_5to8, zeta_cos_1to4);
    Z +=      dot(Z_coef_msin_1to4_ncos_5to8 * theta_final_sin_1to4, zeta_cos_5to8);
    Z +=      dot(Z_coef_msin_5to8_ncos_5to8 * theta_final_sin_5to8, zeta_cos_5to8);

    Z +=      dot(Z_coef_msin_1to4_n0, theta_final_sin_1to4);
    Z +=      dot(Z_coef_msin_5to8_n0, theta_final_sin_5to8);

    Z +=      dot(Z_coef_m0_nsin_1to4, zeta_sin_1to4);
    Z +=      dot(Z_coef_m0_nsin_5to8, zeta_sin_5to8);

    // Compute final distance


    return length(r) - length(vec2(R, Z) - r0);
    //zeta_cos_1to4 * sin(theta_initial)
}

vec3 calcNormal_sdf0( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0);
    // const float eps = 0.0005;
    return normalize( e.xyy*map_sdf0( pos + e.xyy*_stopEpsilon ) + 
					  e.yyx*map_sdf0( pos + e.yyx*_stopEpsilon ) + 
					  e.yxy*map_sdf0( pos + e.yxy*_stopEpsilon ) + 
					  e.xxx*map_sdf0( pos + e.xxx*_stopEpsilon ) );
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


    // raymarch
    int closest_object = -1;
    float closest_dist = _tmax;
    float t = 0.0;
    float h = 0.0;
    vec3 pos;
    
    t = 0.0;
    uint i=0u;
    for(i=0u; i<_maxSteps; i++ )
    {
        pos = p0 + t*cam_norm;
        h = map_sdf0(pos);
        t += h * 0.3;
        if (t>closest_dist) break;
        if( abs(h)<_stopEpsilon ) {
            closest_object = 0;
            closest_dist = t;
        }
    }
    
    vec3 nor;
    if(_mode == 0u) {
        switch(closest_object)
        {
            case -1:
                break;
            
            case 0:
                nor = calcNormal_sdf0(p0 + closest_dist*cam_norm);
                break;
            
        }
    }

    vec4 color = vec4(0.0);
    if( closest_object != -1 ) {
        color.w = 1.0;
        if (_mode == 0u) {
            float dif = clamp( dot(nor,vec3(0.57703)), 0.0, 1.0 ) * 0.2;
            float amb = 1.4 + 0.3*dot(nor,vec3(0.0,1.0,0.0));
            color.xyz = clamp(sdf_colors[closest_object], 0.1, 0.9) * amb + vec3(0.8,0.7,0.5)*dif*0.2;
        } else if(_mode == 1u) {
            color.xyz = mod(pos, 1.0);
        } else if (_mode == 2u) {
            color.xyz=vec3(clamp(float(i) / 255.0, 0.0, 1.0));
        }
    }

    return color;
}

void main() {
    fragColor = mainImage(gl_FragCoord.xy);
}
