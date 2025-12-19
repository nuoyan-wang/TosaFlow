//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;

void forward_node0(
  float v0[10],
  float v1[10],
  float v2[10]
) {	// L29
  #pragma HLS inline
  #pragma HLS resource variable=v0 core=ram_t2p_bram

  #pragma HLS resource variable=v1 core=ram_t2p_bram

  for (int v3 = 0; v3 < 5; v3 += 1) {	// L30
    for (int v4 = 0; v4 < 2; v4 += 1) {	// L31
      #pragma HLS pipeline II=1
      float v5 = v0[(v4 + (v3 * 2))];	// L32
      float v6 = v1[(v4 + (v3 * 2))];	// L33
      float v7 = v5 + v6;	// L34
      v2[(v4 + (v3 * 2))] = v7;	// L35
    }
  }
}

void forward_node2(
  float v8[120],
  float v9[2][2],
  float v10[10],
  int v11,
  int v12
) {	// L40
  #pragma HLS inline
  #pragma HLS resource variable=v8 core=ram_t2p_bram

  #pragma HLS resource variable=v9 core=ram_t2p_bram

  #pragma HLS resource variable=v10 core=ram_t2p_bram

  for (int v13 = 0; v13 < 2; v13 += 1) {	// L41
    for (int v14 = 0; v14 < 2; v14 += 1) {	// L42
      #pragma HLS pipeline II=1
      float v15 = v8[(v13 + (v11 * 2))];	// L43
      float v16 = v9[v13][v14];	// L44
      float v17 = v10[(v14 + (v12 * 2))];	// L45
      float v18 = v15 * v16;	// L46
      float v19 = v17 + v18;	// L47
      v10[(v14 + (v12 * 2))] = v19;	// L48
    }
  }
}

void forward_node3(
  float v20[120][10],
  float v21[2][2],
  int v22,
  int v23
) {	// L53
  #pragma HLS inline
  #pragma HLS resource variable=v21 core=ram_t2p_bram

  for (int v24 = 0; v24 < 2; v24 += 1) {	// L54
    for (int v25 = 0; v25 < 2; v25 += 1) {	// L55
      #pragma HLS pipeline II=1
      float v26 = v20[(v24 + (v22 * 2))][(v25 + (v23 * 2))];	// L56
      v21[v24][v25] = v26;	// L57
    }
  }
}

void forward_node1(
  float v27[120][10],
  float v28[120],
  float v29[10]
) {	// L62
  #pragma HLS resource variable=v28 core=ram_t2p_bram

  #pragma HLS resource variable=v29 core=ram_t2p_bram

  for (int v30 = 0; v30 < 300; v30 += 1) {	// L63
    #pragma HLS dataflow
    int v31 = (v30 % 5);	// L64
    int v32 = (v30 / 5);	// L65
    float v33[2][2];	// L66
    #pragma HLS resource variable=v33 core=ram_t2p_bram

    forward_node3(v27, v33, v32, v31);	// L67
    forward_node2(v28, v33, v29, v32, v31);	// L68
  }
}

void forward_node4(
  float v34[120],
  float v35[120]
) {	// L72
  #pragma HLS inline
  #pragma HLS resource variable=v34 core=ram_t2p_bram

  #pragma HLS resource variable=v35 core=ram_t2p_bram

  for (int v36 = 0; v36 < 60; v36 += 1) {	// L75
    for (int v37 = 0; v37 < 2; v37 += 1) {	// L76
      #pragma HLS pipeline II=1
      float v38 = v34[(v37 + (v36 * 2))];	// L77
      float v39 = min(v38, (float)340282346638528859811704183484516925440.000000);	// L78
      float v40 = max(v39, (float)0.000000);	// L79
      v35[(v37 + (v36 * 2))] = v40;	// L80
    }
  }
}

void forward_node6(
  float v41[2][2],
  float v42[5][5][16],
  float v43[120],
  float v44[120],
  float v45[120],
  int v46,
  int v47,
  int v48,
  int v49
) {	// L85
  #pragma HLS inline
  #pragma HLS resource variable=v41 core=ram_t2p_bram

  #pragma HLS resource variable=v42 core=ram_t2p_bram

  #pragma HLS resource variable=v43 core=ram_t2p_bram

  #pragma HLS resource variable=v44 core=ram_t2p_bram

  #pragma HLS resource variable=v45 core=ram_t2p_bram

  for (int v50 = 0; v50 < 2; v50 += 1) {	// L86
    for (int v51 = 0; v51 < 2; v51 += 1) {	// L87
      #pragma HLS pipeline II=1
      float v52 = v42[v49][v46][(v50 + (v47 * 2))];	// L88
      float v53 = v41[v50][v51];	// L89
      float v54 = v45[(v51 + (v48 * 2))];	// L90
      float v55 = v52 * v53;	// L91
      float v56 = v54 + v55;	// L92
      v45[(v51 + (v48 * 2))] = v56;	// L93
      float v57 = v43[(v51 + (v48 * 2))];	// L94
      float v58 = v57 + v56;	// L95
      if (((-v49) + 4) == 0 && ((-v46) + 4) == 0 && (((-v50) + (v47 * -2)) + 15) == 0) {	// L96
        v44[(v51 + (v48 * 2))] = v58;	// L97
      }
    }
  }
}

void forward_node7(
  float v59[5][5][16][120],
  float v60[2][2],
  int v61,
  int v62,
  int v63,
  int v64
) {	// L103
  #pragma HLS inline
  #pragma HLS resource variable=v60 core=ram_t2p_bram

  for (int v65 = 0; v65 < 2; v65 += 1) {	// L104
    for (int v66 = 0; v66 < 2; v66 += 1) {	// L105
      #pragma HLS pipeline II=1
      float v67 = v59[v61][v62][(v65 + (v63 * 2))][(v66 + (v64 * 2))];	// L106
      v60[v65][v66] = v67;	// L107
    }
  }
}

void forward_node5(
  float v68[120],
  float v69[5][5][16],
  float v70[5][5][16][120],
  float v71[120]
) {	// L112
  #pragma HLS resource variable=v68 core=ram_t2p_bram

  #pragma HLS resource variable=v69 core=ram_t2p_bram

  #pragma HLS resource variable=v71 core=ram_t2p_bram

  float v72[120];	// L113
  #pragma HLS resource variable=v72 core=ram_t2p_bram

  for (int v73 = 0; v73 < 12000; v73 += 1) {	// L114
    #pragma HLS dataflow
    int v74 = (v73 % 60);	// L115
    int v75 = ((v73 / 60) % 8);	// L116
    int v76 = (((v73 / 60) / 8) % 5);	// L117
    int v77 = (((v73 / 60) / 8) / 5);	// L118
    float v78[2][2];	// L119
    #pragma HLS resource variable=v78 core=ram_t2p_bram

    forward_node7(v70, v78, v77, v76, v75, v74);	// L120
    forward_node6(v78, v69, v68, v71, v72, v76, v75, v74, v77);	// L121
  }
}

void forward_node9(
  float v79[2],
  float v80[5][5][16],
  int v81,
  int v82,
  int v83
) {	// L125
  #pragma HLS inline
  #pragma HLS resource variable=v79 core=ram_t2p_bram

  #pragma HLS resource variable=v80 core=ram_t2p_bram

  for (int v84 = 0; v84 < 2; v84 += 1) {	// L126
    #pragma HLS pipeline II=1
    float v85 = v79[v84];	// L127
    float v86 = v80[v81][v82][(v84 + (v83 * 2))];	// L128
    float v87 = max(v86, v85);	// L129
    v80[v81][v82][(v84 + (v83 * 2))] = v87;	// L130
  }
}

void forward_node10(
  float v88[10][10][16],
  float v89[2],
  int v90,
  int v91,
  int v92,
  int v93,
  int v94
) {	// L134
  #pragma HLS inline
  #pragma HLS resource variable=v89 core=ram_t2p_bram

  for (int v95 = 0; v95 < 2; v95 += 1) {	// L135
    #pragma HLS pipeline II=1
    float v96 = v88[((v90 * 2) + v91)][((v92 * 2) + v93)][(v95 + (v94 * 2))];	// L136
    v89[v95] = v96;	// L137
  }
}

void forward_node8(
  hls::stream<bool> &v97,
  float v98[10][10][16],
  float v99[5][5][16]
) {	// L141
  #pragma HLS resource variable=v99 core=ram_t2p_bram

  v97.read();	// L142
  for (int v100 = 0; v100 < 800; v100 += 1) {	// L143
    #pragma HLS dataflow
    int v101 = (v100 % 8);	// L144
    int v102 = ((v100 / 8) % 5);	// L145
    int v103 = (((v100 / 8) / 5) % 5);	// L146
    int v104 = ((((v100 / 8) / 5) / 5) % 2);	// L147
    int v105 = ((((v100 / 8) / 5) / 5) / 2);	// L148
    float v106[2];	// L149
    #pragma HLS resource variable=v106 core=ram_t2p_bram

    forward_node10(v98, v106, v103, v105, v102, v104, v101);	// L150
    forward_node9(v106, v99, v103, v102, v101);	// L151
  }
}

void forward_node12(
  float v107[2][2][2],
  float v108[10][10][16],
  int v109,
  int v110,
  int v111
) {	// L155
  #pragma HLS inline
  #pragma HLS resource variable=v107 core=ram_t2p_bram

  for (int v112 = 0; v112 < 2; v112 += 1) {	// L156
    for (int v113 = 0; v113 < 2; v113 += 1) {	// L157
      for (int v114 = 0; v114 < 2; v114 += 1) {	// L158
        #pragma HLS pipeline II=1
        float v115 = v107[v112][v113][v114];	// L159
        v108[(v112 + (v109 * 2))][(v113 + (v110 * 2))][(v114 + (v111 * 2))] = v115;	// L160
      }
    }
  }
}

void forward_node13(
  float v116[2][2][2],
  float v117[10][10][16],
  int v118,
  int v119,
  int v120
) {	// L166
  #pragma HLS inline
  #pragma HLS resource variable=v116 core=ram_t2p_bram

  for (int v121 = 0; v121 < 2; v121 += 1) {	// L167
    for (int v122 = 0; v122 < 2; v122 += 1) {	// L168
      for (int v123 = 0; v123 < 2; v123 += 1) {	// L169
        #pragma HLS pipeline II=1
        float v124 = v116[v121][v122][v123];	// L170
        v117[(v121 + (v118 * 2))][(v122 + (v119 * 2))][(v123 + (v120 * 2))] = v124;	// L171
      }
    }
  }
}

void forward_node14(
  float v125[2][2][2],
  float v126[16],
  float v127[2][2],
  float v128[2][2][2],
  float v129[2][2][2],
  float v130[2][2][2],
  int v131,
  int v132,
  int v133,
  int v134
) {	// L177
  #pragma HLS inline
  #pragma HLS resource variable=v125 core=ram_t2p_bram

  #pragma HLS resource variable=v126 core=ram_t2p_bram

  #pragma HLS resource variable=v127 core=ram_t2p_bram

  #pragma HLS resource variable=v128 core=ram_t2p_bram

  #pragma HLS resource variable=v129 core=ram_t2p_bram

  #pragma HLS resource variable=v130 core=ram_t2p_bram

  for (int v135 = 0; v135 < 2; v135 += 1) {	// L180
    for (int v136 = 0; v136 < 2; v136 += 1) {	// L181
      for (int v137 = 0; v137 < 2; v137 += 1) {	// L182
        for (int v138 = 0; v138 < 2; v138 += 1) {	// L183
          #pragma HLS pipeline II=1
          float v139 = v125[v136][v137][v135];	// L184
          float v140 = v127[v135][v138];	// L185
          float v141 = v128[v136][v137][v138];	// L186
          float v142 = v130[v136][v137][v138];	// L187
          float v143 = (v135 == 0) ? v141 : v142;	// L188
          float v144 = v139 * v140;	// L189
          float v145 = v143 + v144;	// L190
          v130[v136][v137][v138] = v145;	// L191
          float v146 = v126[(v138 + (v132 * 2))];	// L192
          float v147 = v146 + v145;	// L193
          float v148 = min(v147, (float)340282346638528859811704183484516925440.000000);	// L194
          float v149 = max(v148, (float)0.000000);	// L195
          if (((-v133) + 4) == 0 && ((-v134) + 4) == 0 && (((-v135) + (v131 * -2)) + 5) == 0) {	// L196
            v129[v136][v137][v138] = v149;	// L197
          }
        }
      }
    }
  }
}

void forward_node15(
  float v150[10][10][16],
  float v151[2][2][2],
  int v152,
  int v153,
  int v154
) {	// L205
  #pragma HLS inline
  #pragma HLS resource variable=v151 core=ram_t2p_bram

  for (int v155 = 0; v155 < 2; v155 += 1) {	// L206
    for (int v156 = 0; v156 < 2; v156 += 1) {	// L207
      for (int v157 = 0; v157 < 2; v157 += 1) {	// L208
        #pragma HLS pipeline II=1
        float v158 = v150[(v155 + (v152 * 2))][(v156 + (v153 * 2))][(v157 + (v154 * 2))];	// L209
        v151[v155][v156][v157] = v158;	// L210
      }
    }
  }
}

void forward_node16(
  float v159[5][5][6][16],
  float v160[2][2],
  int v161,
  int v162,
  int v163,
  int v164
) {	// L216
  #pragma HLS inline
  #pragma HLS resource variable=v160 core=ram_t2p_bram

  for (int v165 = 0; v165 < 2; v165 += 1) {	// L217
    for (int v166 = 0; v166 < 2; v166 += 1) {	// L218
      #pragma HLS pipeline II=1
      float v167 = v159[v161][v162][(v165 + (v163 * 2))][(v166 + (v164 * 2))];	// L219
      v160[v165][v166] = v167;	// L220
    }
  }
}

void forward_node17(
  float v168[14][14][6],
  float v169[2][2][2],
  int v170,
  int v171,
  int v172,
  int v173,
  int v174
) {	// L225
  #pragma HLS inline
  #pragma HLS resource variable=v169 core=ram_t2p_bram

  for (int v175 = 0; v175 < 2; v175 += 1) {	// L226
    for (int v176 = 0; v176 < 2; v176 += 1) {	// L227
      for (int v177 = 0; v177 < 2; v177 += 1) {	// L228
        #pragma HLS pipeline II=1
        float v178 = v168[((v175 + v170) + (v171 * 2))][((v176 + v172) + (v173 * 2))][(v177 + (v174 * 2))];	// L229
        v169[v175][v176][v177] = v178;	// L230
      }
    }
  }
}

void forward_node11(
  float v179[5][5][6][16],
  float v180[16],
  hls::stream<bool> &v181,
  float v182[14][14][6],
  float v183[10][10][16],
  hls::stream<bool> &v184,
  float v185[10][10][16],
  float v186[10][10][16]
) {	// L236
  #pragma HLS resource variable=v180 core=ram_t2p_bram

  v181.read();	// L238
  for (int v187 = 0; v187 < 15000; v187 += 1) {	// L239
    #pragma HLS dataflow
    int v188 = (v187 % 8);	// L240
    int v189 = ((v187 / 8) % 5);	// L241
    int v190 = (((v187 / 8) / 5) % 5);	// L242
    int v191 = ((((v187 / 8) / 5) / 5) % 3);	// L243
    int v192 = (((((v187 / 8) / 5) / 5) / 3) % 5);	// L244
    int v193 = (((((v187 / 8) / 5) / 5) / 3) / 5);	// L245
    float v194[2][2][2];	// L246
    #pragma HLS resource variable=v194 core=ram_t2p_bram

    float v195[2][2][2];	// L247
    #pragma HLS resource variable=v195 core=ram_t2p_bram

    float v196[2][2];	// L248
    #pragma HLS resource variable=v196 core=ram_t2p_bram

    float v197[2][2][2];	// L249
    #pragma HLS resource variable=v197 core=ram_t2p_bram

    forward_node17(v182, v197, v193, v190, v192, v189, v191);	// L250
    forward_node16(v179, v196, v193, v192, v191, v188);	// L251
    forward_node15(v183, v195, v190, v189, v188);	// L252
    float v198[2][2][2];	// L253
    #pragma HLS resource variable=v198 core=ram_t2p_bram

    forward_node14(v197, v180, v196, v195, v194, v198, v191, v188, v193, v192);	// L254
    forward_node13(v198, v186, v190, v189, v188);	// L255
    forward_node12(v194, v185, v190, v189, v188);	// L256
  }
  v184.write(true);	// L258
}

void forward_node19(
  float v199[2][2][2],
  float v200[14][14][6],
  int v201,
  int v202,
  int v203
) {	// L261
  #pragma HLS inline
  #pragma HLS resource variable=v199 core=ram_t2p_bram

  for (int v204 = 0; v204 < 2; v204 += 1) {	// L262
    for (int v205 = 0; v205 < 2; v205 += 1) {	// L263
      for (int v206 = 0; v206 < 2; v206 += 1) {	// L264
        #pragma HLS pipeline II=1
        float v207 = v199[v204][v205][v206];	// L265
        v200[(v204 + (v201 * 2))][(v205 + (v202 * 2))][(v206 + (v203 * 2))] = v207;	// L266
      }
    }
  }
}

void forward_node20(
  float v208[2][2][2],
  float v209[2][2][2],
  float v210[2][2][2]
) {	// L272
  #pragma HLS inline
  #pragma HLS resource variable=v208 core=ram_t2p_bram

  #pragma HLS resource variable=v209 core=ram_t2p_bram

  #pragma HLS resource variable=v210 core=ram_t2p_bram

  for (int v211 = 0; v211 < 2; v211 += 1) {	// L273
    for (int v212 = 0; v212 < 2; v212 += 1) {	// L274
      for (int v213 = 0; v213 < 2; v213 += 1) {	// L275
        #pragma HLS pipeline II=1
        float v214 = v208[v211][v212][v213];	// L276
        float v215 = v209[v211][v212][v213];	// L277
        float v216 = max(v215, v214);	// L278
        v210[v211][v212][v213] = v216;	// L279
      }
    }
  }
}

void forward_node21(
  float v217[14][14][6],
  float v218[2][2][2],
  int v219,
  int v220,
  int v221
) {	// L285
  #pragma HLS inline
  #pragma HLS resource variable=v218 core=ram_t2p_bram

  for (int v222 = 0; v222 < 2; v222 += 1) {	// L286
    for (int v223 = 0; v223 < 2; v223 += 1) {	// L287
      for (int v224 = 0; v224 < 2; v224 += 1) {	// L288
        #pragma HLS pipeline II=1
        float v225 = v217[(v222 + (v219 * 2))][(v223 + (v220 * 2))][(v224 + (v221 * 2))];	// L289
        v218[v222][v223][v224] = v225;	// L290
      }
    }
  }
}

void forward_node22(
  float v226[28][28][6],
  float v227[2][2][2],
  int v228,
  int v229,
  int v230,
  int v231,
  int v232
) {	// L296
  #pragma HLS inline
  #pragma HLS resource variable=v227 core=ram_t2p_bram

  for (int v233 = 0; v233 < 2; v233 += 1) {	// L297
    for (int v234 = 0; v234 < 2; v234 += 1) {	// L298
      for (int v235 = 0; v235 < 2; v235 += 1) {	// L299
        #pragma HLS pipeline II=1
        float v236 = v226[(((v233 * 2) + v228) + (v229 * 4))][(((v234 * 2) + v230) + (v231 * 4))][(v235 + (v232 * 2))];	// L300
        v227[v233][v234][v235] = v236;	// L301
      }
    }
  }
}

void forward_node18(
  hls::stream<bool> &v237,
  float v238[28][28][6],
  float v239[14][14][6],
  hls::stream<bool> &v240,
  float v241[14][14][6]
) {	// L307
  v237.read();	// L309
  for (int v242 = 0; v242 < 588; v242 += 1) {	// L310
    #pragma HLS dataflow
    int v243 = (v242 % 3);	// L311
    int v244 = ((v242 / 3) % 7);	// L312
    int v245 = (((v242 / 3) / 7) % 7);	// L313
    int v246 = ((((v242 / 3) / 7) / 7) % 2);	// L314
    int v247 = ((((v242 / 3) / 7) / 7) / 2);	// L315
    float v248[2][2][2];	// L316
    #pragma HLS resource variable=v248 core=ram_t2p_bram

    float v249[2][2][2];	// L317
    #pragma HLS resource variable=v249 core=ram_t2p_bram

    forward_node22(v238, v249, v247, v245, v246, v244, v243);	// L318
    forward_node21(v239, v248, v245, v244, v243);	// L319
    float v250[2][2][2];	// L320
    #pragma HLS resource variable=v250 core=ram_t2p_bram

    forward_node20(v249, v248, v250);	// L321
    forward_node19(v250, v241, v245, v244, v243);	// L322
  }
  v240.write(true);	// L324
}

void forward_node24(
  float v251[2][2][2],
  float v252[28][28][6],
  int v253,
  int v254,
  int v255
) {	// L327
  #pragma HLS inline
  #pragma HLS resource variable=v251 core=ram_t2p_bram

  for (int v256 = 0; v256 < 2; v256 += 1) {	// L328
    for (int v257 = 0; v257 < 2; v257 += 1) {	// L329
      for (int v258 = 0; v258 < 2; v258 += 1) {	// L330
        #pragma HLS pipeline II=1
        float v259 = v251[v256][v257][v258];	// L331
        v252[(v256 + (v253 * 2))][(v257 + (v254 * 2))][(v258 + (v255 * 2))] = v259;	// L332
      }
    }
  }
}

void forward_node25(
  float v260[2][2][2],
  float v261[28][28][6],
  int v262,
  int v263,
  int v264
) {	// L338
  #pragma HLS inline
  #pragma HLS resource variable=v260 core=ram_t2p_bram

  for (int v265 = 0; v265 < 2; v265 += 1) {	// L339
    for (int v266 = 0; v266 < 2; v266 += 1) {	// L340
      for (int v267 = 0; v267 < 2; v267 += 1) {	// L341
        #pragma HLS pipeline II=1
        float v268 = v260[v265][v266][v267];	// L342
        v261[(v265 + (v262 * 2))][(v266 + (v263 * 2))][(v267 + (v264 * 2))] = v268;	// L343
      }
    }
  }
}

void forward_node26(
  float v269[6],
  float v270[5][5][6],
  float v271[2][2],
  float v272[2][2][2],
  float v273[2][2][2],
  float v274[2][2][2],
  int v275,
  int v276,
  int v277
) {	// L349
  #pragma HLS inline
  #pragma HLS resource variable=v269 core=ram_t2p_bram

  #pragma HLS resource variable=v270 core=ram_t2p_bram

  #pragma HLS resource variable=v271 core=ram_t2p_bram

  #pragma HLS resource variable=v272 core=ram_t2p_bram

  #pragma HLS resource variable=v273 core=ram_t2p_bram

  #pragma HLS resource variable=v274 core=ram_t2p_bram

  for (int v278 = 0; v278 < 2; v278 += 1) {	// L352
    for (int v279 = 0; v279 < 2; v279 += 1) {	// L353
      for (int v280 = 0; v280 < 2; v280 += 1) {	// L354
        #pragma HLS pipeline II=1
        float v281 = v271[v278][v279];	// L355
        float v282 = v270[v276][v275][(v280 + (v277 * 2))];	// L356
        float v283 = v272[v278][v279][v280];	// L357
        float v284 = v281 * v282;	// L358
        float v285 = v283 + v284;	// L359
        v274[v278][v279][v280] = v285;	// L360
        float v286 = v269[(v280 + (v277 * 2))];	// L361
        float v287 = v286 + v285;	// L362
        float v288 = min(v287, (float)340282346638528859811704183484516925440.000000);	// L363
        float v289 = max(v288, (float)0.000000);	// L364
        if (((-v276) + 4) == 0 && ((-v275) + 4) == 0) {	// L365
          v273[v278][v279][v280] = v289;	// L366
        }
      }
    }
  }
}

void forward_node27(
  float v290[28][28][6],
  float v291[2][2][2],
  int v292,
  int v293,
  int v294
) {	// L373
  #pragma HLS inline
  #pragma HLS resource variable=v291 core=ram_t2p_bram

  for (int v295 = 0; v295 < 2; v295 += 1) {	// L374
    for (int v296 = 0; v296 < 2; v296 += 1) {	// L375
      for (int v297 = 0; v297 < 2; v297 += 1) {	// L376
        #pragma HLS pipeline II=1
        float v298 = v290[(v295 + (v292 * 2))][(v296 + (v293 * 2))][(v297 + (v294 * 2))];	// L377
        v291[v295][v296][v297] = v298;	// L378
      }
    }
  }
}

void forward_node28(
  float v299[32][32],
  float v300[2][2],
  int v301,
  int v302,
  int v303,
  int v304
) {	// L384
  #pragma HLS inline
  #pragma HLS resource variable=v300 core=ram_t2p_bram

  for (int v305 = 0; v305 < 2; v305 += 1) {	// L385
    for (int v306 = 0; v306 < 2; v306 += 1) {	// L386
      #pragma HLS pipeline II=1
      float v307 = v299[((v305 + v301) + (v302 * 2))][((v306 + v303) + (v304 * 2))];	// L387
      v300[v305][v306] = v307;	// L388
    }
  }
}

void forward_node23(
  float v308[32][32],
  float v309[5][5][6],
  float v310[6],
  float v311[28][28][6],
  float v312[28][28][6],
  hls::stream<bool> &v313,
  float v314[28][28][6]
) {	// L393
  #pragma HLS resource variable=v309 core=ram_t2p_bram

  #pragma HLS resource variable=v310 core=ram_t2p_bram

  for (int v315 = 0; v315 < 14700; v315 += 1) {	// L395
    #pragma HLS dataflow
    int v316 = (v315 % 3);	// L396
    int v317 = ((v315 / 3) % 14);	// L397
    int v318 = (((v315 / 3) / 14) % 14);	// L398
    int v319 = ((((v315 / 3) / 14) / 14) % 5);	// L399
    int v320 = ((((v315 / 3) / 14) / 14) / 5);	// L400
    float v321[2][2][2];	// L401
    #pragma HLS resource variable=v321 core=ram_t2p_bram

    float v322[2][2][2];	// L402
    #pragma HLS resource variable=v322 core=ram_t2p_bram

    float v323[2][2];	// L403
    #pragma HLS resource variable=v323 core=ram_t2p_bram

    forward_node28(v308, v323, v320, v318, v319, v317);	// L404
    forward_node27(v311, v322, v318, v317, v316);	// L405
    float v324[2][2][2];	// L406
    #pragma HLS resource variable=v324 core=ram_t2p_bram

    forward_node26(v310, v309, v323, v322, v321, v324, v319, v320, v316);	// L407
    forward_node25(v324, v312, v318, v317, v316);	// L408
    forward_node24(v321, v314, v318, v317, v316);	// L409
  }
  v313.write(true);	// L411
}

/// This is top function.
void forward(
  float v325[32][32],
  float v326[10],
  float v327[5][5][16][120],
  float v328[5][5][6][16],
  float v329[120][10],
  float v330[28][28][6],
  float v331[28][28][6],
  float v332[28][28][6],
  float v333[28][28][6],
  float v334[14][14][6],
  float v335[14][14][6],
  float v336[14][14][6],
  float v337[10][10][16],
  float v338[10][10][16],
  float v339[10][10][16],
  float v340[10][10][16]
) {	// L414
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS dataflow

  #pragma HLS interface ap_memory port=v340
  #pragma HLS stable variable=v340

  #pragma HLS interface ap_memory port=v339
  #pragma HLS stable variable=v339

  #pragma HLS interface ap_memory port=v338
  #pragma HLS stable variable=v338

  #pragma HLS interface ap_memory port=v337
  #pragma HLS stable variable=v337

  #pragma HLS interface ap_memory port=v336
  #pragma HLS stable variable=v336

  #pragma HLS interface ap_memory port=v335
  #pragma HLS stable variable=v335

  #pragma HLS interface ap_memory port=v334
  #pragma HLS stable variable=v334

  #pragma HLS interface ap_memory port=v333
  #pragma HLS stable variable=v333

  #pragma HLS interface ap_memory port=v332
  #pragma HLS stable variable=v332

  #pragma HLS interface ap_memory port=v331
  #pragma HLS stable variable=v331

  #pragma HLS interface ap_memory port=v330
  #pragma HLS stable variable=v330

  #pragma HLS interface ap_memory port=v329
  #pragma HLS stable variable=v329

  #pragma HLS interface ap_memory port=v328
  #pragma HLS stable variable=v328

  #pragma HLS interface ap_memory port=v327
  #pragma HLS stable variable=v327

  #pragma HLS interface ap_memory port=v326
  #pragma HLS stable variable=v326

  #pragma HLS interface ap_memory port=v325
  #pragma HLS stable variable=v325

  float v357[10] = {(float)0.039555, (float)0.023299, (float)0.072903, (float)0.021692, (float)0.104451, (float)0.090518, (float)-0.003619, (float)0.073619, (float)0.092954, (float)0.008757};	// L447
  #pragma HLS resource variable=v357 core=ram_t2p_bram

  float v358[5][5][6] = {(float)0.054577, (float)-0.015495, (float)0.174924, (float)0.026375, (float)-0.176736, (float)0.089606, (float)0.239890, (float)0.099448, (float)0.134596, (float)-0.181893, (float)-0.151720, (float)0.071561, (float)0.046091, (float)0.011086, (float)-0.187919, (float)-0.164046, (float)0.058466, (float)0.097658, (float)0.189254, (float)-0.067664, (float)0.046181, (float)0.182536, (float)0.140155, (float)-0.105534, (float)-0.059370, (float)0.079366, (float)-0.165645, (float)0.130094, (float)-0.108961, (float)0.137008, (float)-0.113319, (float)0.040238, (float)0.009779, (float)0.221941, (float)0.150772, (float)0.033139, (float)-0.064101, (float)0.150162, (float)-0.021289, (float)0.057020, (float)-0.084157, (float)-0.137870, (float)0.116482, (float)0.043000, (float)-0.048410, (float)0.188594, (float)-0.075401, (float)0.054783, (float)-0.149269, (float)0.094292, (float)0.065764, (float)0.011071, (float)-0.008689, (float)-0.155438, (float)-0.017120, (float)-0.007492, (float)-0.165713, (float)-0.008246, (float)0.034697, (float)0.048253, (float)-0.131366, (float)-0.005065, (float)-0.093086, (float)-0.037462, (float)-0.162376, (float)0.119975, (float)-0.311866, (float)-0.112828, (float)0.122882, (float)0.119720, (float)0.031016, (float)0.174484, (float)-0.210143, (float)0.120868, (float)0.135085, (float)-0.015477, (float)0.031226, (float)-0.181807, (float)-0.069568, (float)-0.080619, (float)0.039021, (float)-0.074788, (float)0.046656, (float)0.014725, (float)0.028491, (float)0.042296, (float)-0.034685, (float)-0.088741, (float)-0.115986, (float)0.115346, (float)0.039154, (float)-0.010737, (float)0.008144, (float)0.060459, (float)0.170055, (float)0.161037, (float)-0.025997, (float)-0.054286, (float)0.093855, (float)0.079697, (float)0.039935, (float)-0.001829, (float)0.094974, (float)0.086668, (float)0.042818, (float)0.220836, (float)-0.078396, (float)-0.143296, (float)0.010454, (float)0.122190, (float)0.099970, (float)-0.082229, (float)0.086352, (float)-0.030076, (float)0.077851, (float)-0.100322, (float)-0.108910, (float)-0.006711, (float)-0.079305, (float)-0.147059, (float)0.168553, (float)-0.174280, (float)-0.006766, (float)0.243419, (float)-0.135728, (float)0.166156, (float)0.104512, (float)-0.050791, (float)-0.127050, (float)0.012329, (float)0.025542, (float)0.085518, (float)0.041096, (float)-0.069104, (float)0.073929, (float)-0.065600, (float)-0.016217, (float)-0.066053, (float)0.014644, (float)0.141563, (float)0.175849, (float)-0.066764, (float)0.181592, (float)-0.050629, (float)-0.002489, (float)0.102478, (float)0.156797, (float)0.079085, (float)0.099328, (float)-0.000406};	// L448
  #pragma HLS resource variable=v358 core=ram_t2p_bram

  float v359[6] = {(float)-0.093682, (float)-0.083960, (float)-0.052419, (float)-0.070656, (float)-0.001334, (float)-0.073384};	// L449
  #pragma HLS resource variable=v359 core=ram_t2p_bram

  float v360[16] = {(float)-0.096252, (float)-0.078570, (float)-0.080000, (float)-0.006678, (float)-0.036978, (float)-0.040000, (float)-0.043175, (float)-0.054606, (float)-0.003842, (float)-0.006323, (float)0.012949, (float)0.071626, (float)0.026551, (float)0.071407, (float)0.345437, (float)-0.079999};	// L450
  #pragma HLS resource variable=v360 core=ram_t2p_bram

  float v361[120] = {(float)-0.040000, (float)-0.020977, (float)-0.025260, (float)-0.011291, (float)-0.040000, (float)-0.040000, (float)-0.007104, (float)0.123800, (float)-0.040000, (float)-0.126483, (float)0.175910, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)0.000053, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)-0.006798, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)0.147117, (float)-0.040000, (float)-0.003974, (float)-0.005638, (float)0.005124, (float)-0.045583, (float)-0.040000, (float)-0.040000, (float)0.012136, (float)-0.040000, (float)-0.013427, (float)-0.032908, (float)0.051463, (float)-0.040000, (float)-0.040000, (float)-0.027865, (float)-0.040000, (float)-0.030247, (float)0.002224, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)0.017196, (float)0.052518, (float)0.277972, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)-0.020854, (float)-0.040000, (float)0.086088, (float)0.002074, (float)0.120874, (float)-0.040000, (float)0.018952, (float)-0.040000, (float)-0.039999, (float)-0.049238, (float)0.001968, (float)-0.040000, (float)0.099777, (float)-0.040000, (float)-0.040000, (float)-0.000313, (float)-0.042707, (float)0.047646, (float)-0.040000, (float)0.022962, (float)-0.040000, (float)-0.040000, (float)0.128993, (float)-0.079715, (float)0.083239, (float)-0.000398, (float)-0.040000, (float)0.001765, (float)0.185148, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)0.198363, (float)0.022491, (float)0.014955, (float)-0.040000, (float)-0.040000, (float)-0.006487, (float)-0.006388, (float)-0.040000, (float)-0.020015, (float)-0.040000, (float)-0.040000, (float)0.012344, (float)-0.040000, (float)0.050312, (float)0.044808, (float)0.008754, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)-0.001461, (float)-0.001090, (float)-0.040000, (float)-0.040000, (float)0.030712, (float)0.023594, (float)0.020151, (float)-0.005048, (float)-0.040000, (float)-0.040000, (float)-0.040000, (float)-0.043701, (float)-0.040000, (float)-0.021247, (float)0.021379, (float)0.034406, (float)-0.040000};	// L451
  #pragma HLS resource variable=v361 core=ram_t2p_bram

  hls::stream<bool> v362;	// L452
  forward_node23(v325, v358, v359, v333, v332, v362, v330);	// L453
  hls::stream<bool> v363;	// L454
  forward_node18(v362, v331, v335, v363, v334);	// L455
  hls::stream<bool> v364;	// L456
  forward_node11(v328, v360, v363, v336, v340, v364, v337, v339);	// L457
  float v365[5][5][16];	// L458
  #pragma HLS resource variable=v365 core=ram_t2p_bram

  forward_node8(v364, v338, v365);	// L459
  float v366[120];	// L460
  #pragma HLS resource variable=v366 core=ram_t2p_bram

  forward_node5(v361, v365, v327, v366);	// L461
  float v367[120];	// L462
  #pragma HLS resource variable=v367 core=ram_t2p_bram

  forward_node4(v366, v367);	// L463
  float v368[10];	// L464
  #pragma HLS resource variable=v368 core=ram_t2p_bram

  forward_node1(v329, v367, v368);	// L465
  forward_node0(v368, v357, v326);	// L466
}