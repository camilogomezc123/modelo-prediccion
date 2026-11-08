# Búsqueda automática: salida temprana de UCI desde el primer día

Generado 2026-10-30 07:28. Elección por AUC en CV (entrenamiento < 2026-02-01); la prueba temporal (feb-ago 2026) solo se informa.

## Sale de UCI en 2 días o menos

Tasa: 27%. Elegido: **catboost** con variables **base** — AUC CV 0.693, AUC prueba 0.701 (IC95 0.663-0.735). Corte ALTA probabilidad de salida temprana: p ≥ 0.27.

### Las 10 mejores combinaciones (por AUC CV)

| objetivo      | conjunto          | familia           |   auc_cv |   auc_prueba | ic95_prueba   |   ap_prueba |   brier_prueba |   intentos |
|:--------------|:------------------|:------------------|---------:|-------------:|:--------------|------------:|---------------:|-----------:|
| salida_uci_2d | base              | catboost          |   0.6933 |       0.7008 | 0.663-0.735   |      0.4904 |         0.1609 |         26 |
| salida_uci_2d | base              | ensamble_promedio |   0.6919 |       0.7035 | 0.668-0.737   |      0.4908 |         0.1634 |          0 |
| salida_uci_2d | base              | ensamble_apilado  |   0.691  |       0.703  | 0.666-0.736   |      0.4913 |         0.1603 |          0 |
| salida_uci_2d | completo          | catboost          |   0.6905 |       0.6988 | 0.663-0.730   |      0.4902 |         0.1615 |         28 |
| salida_uci_2d | completo          | lightgbm          |   0.6898 |       0.688  | 0.654-0.719   |      0.4714 |         0.1641 |        200 |
| salida_uci_2d | completo          | xgboost           |   0.689  |       0.6974 | 0.661-0.729   |      0.4815 |         0.1624 |        103 |
| salida_uci_2d | base+historia+dx3 | extra_trees       |   0.6889 |       0.6949 | 0.660-0.728   |      0.4738 |         0.2055 |         36 |
| salida_uci_2d | base+historia+dx3 | lightgbm          |   0.6885 |       0.7008 | 0.664-0.732   |      0.4839 |         0.1616 |        200 |
| salida_uci_2d | base+historia+dx3 | catboost          |   0.6885 |       0.7007 | 0.662-0.732   |      0.491  |         0.1611 |         22 |
| salida_uci_2d | completo          | extra_trees       |   0.6884 |       0.6913 | 0.656-0.727   |      0.4697 |         0.205  |         28 |

### Mejor AUC CV por conjunto de variables

| conjunto          |   auc_cv |   auc_prueba |
|:------------------|---------:|-------------:|
| base              |   0.6933 |       0.7069 |
| base+historia     |   0.6874 |       0.7059 |
| base+historia+dx3 |   0.6889 |       0.7017 |
| completo          |   0.6905 |       0.6998 |

### Importancia de variables (permutación, prueba)

|                                   |   pérdida de AUC |
|:----------------------------------|-----------------:|
| puntuacion_sofa_inicial           |           0.0766 |
| aseguradora                       |           0.0264 |
| barthel_inicial                   |           0.017  |
| capitulo_dx                       |           0.0151 |
| procedencia_piso                  |           0.0134 |
| dxp_arritmia                      |           0.0065 |
| dxp_desnutricion                  |           0.0049 |
| edad                              |           0.004  |
| dxp_acv                           |           0.0014 |
| dxp_neumonia                      |           0.0008 |
| dxp_obstetrico                    |           0.0005 |
| dxp_falla_cardiaca                |           0.0004 |
| dxp_valvulopatia                  |           0.0004 |
| dxp_hemorragia_digestiva          |           0.0001 |
| dxp_erc                           |           0.0001 |
| dxp_diabetes                      |           0.0001 |
| dxp_falla_respiratoria            |           0      |
| dxp_cancer                        |          -0.0001 |
| dxp_hta                           |          -0.0001 |
| dxp_trauma                        |          -0.0004 |
| dxp_sepsis                        |          -0.0005 |
| dxp_iam                           |          -0.0007 |
| dxp_cardiopatia_isquemica_cronica |          -0.0007 |

## Sale de UCI en 3 días o menos

Tasa: 40%. Elegido: **catboost** con variables **base+historia+dx3** — AUC CV 0.686, AUC prueba 0.679 (IC95 0.646-0.709). Corte ALTA probabilidad de salida temprana: p ≥ 0.42.

### Las 10 mejores combinaciones (por AUC CV)

| objetivo      | conjunto          | familia           |   auc_cv |   auc_prueba | ic95_prueba   |   ap_prueba |   brier_prueba |   intentos |
|:--------------|:------------------|:------------------|---------:|-------------:|:--------------|------------:|---------------:|-----------:|
| salida_uci_3d | base+historia+dx3 | catboost          |   0.6862 |       0.6787 | 0.646-0.709   |      0.5756 |         0.2078 |         19 |
| salida_uci_3d | completo          | catboost          |   0.6862 |       0.6698 | 0.637-0.700   |      0.5647 |         0.2105 |         19 |
| salida_uci_3d | base+historia+dx3 | ensamble_promedio |   0.6857 |       0.6799 | 0.649-0.710   |      0.5729 |         0.2102 |          0 |
| salida_uci_3d | base+historia+dx3 | ensamble_apilado  |   0.685  |       0.6796 | 0.648-0.710   |      0.5731 |         0.2079 |          0 |
| salida_uci_3d | base+historia+dx3 | lightgbm          |   0.6842 |       0.681  | 0.651-0.709   |      0.5716 |         0.2077 |        200 |
| salida_uci_3d | completo          | lightgbm          |   0.6837 |       0.6694 | 0.638-0.698   |      0.5553 |         0.2115 |        200 |
| salida_uci_3d | base+historia+dx3 | xgboost           |   0.6835 |       0.685  | 0.655-0.716   |      0.576  |         0.2069 |         91 |
| salida_uci_3d | completo          | xgboost           |   0.683  |       0.6778 | 0.647-0.708   |      0.5668 |         0.2095 |         61 |
| salida_uci_3d | base              | catboost          |   0.6821 |       0.6735 | 0.642-0.703   |      0.5712 |         0.2088 |         23 |
| salida_uci_3d | base+historia+dx3 | random_forest     |   0.6813 |       0.6766 | 0.645-0.705   |      0.5708 |         0.2184 |         46 |

### Mejor AUC CV por conjunto de variables

| conjunto          |   auc_cv |   auc_prueba |
|:------------------|---------:|-------------:|
| base              |   0.6821 |       0.6853 |
| base+historia     |   0.6804 |       0.6831 |
| base+historia+dx3 |   0.6862 |       0.685  |
| completo          |   0.6862 |       0.6814 |

### Importancia de variables (permutación, prueba)

|                                   |   pérdida de AUC |
|:----------------------------------|-----------------:|
| puntuacion_sofa_inicial           |           0.0732 |
| dx3                               |           0.0271 |
| aseguradora                       |           0.016  |
| barthel_inicial                   |           0.0083 |
| capitulo_dx                       |           0.0018 |
| procedencia_piso                  |           0.0013 |
| dxp_arritmia                      |           0.0013 |
| dxp_acv                           |           0.0007 |
| dxp_desnutricion                  |           0.0005 |
| dxp_obstetrico                    |           0.0003 |
| dxp_trauma                        |           0.0002 |
| dxp_falla_cardiaca                |           0.0001 |
| dxp_sepsis                        |           0.0001 |
| dxp_diabetes                      |           0.0001 |
| dxp_erc                           |           0.0001 |
| dxp_hemorragia_digestiva          |           0.0001 |
| dxp_valvulopatia                  |           0.0001 |
| dxp_hta                           |           0      |
| dxp_falla_respiratoria            |           0      |
| hist_reingreso_30d                |           0      |
| dxp_cardiopatia_isquemica_cronica |          -0.0001 |
| dxp_iam                           |          -0.0001 |
| hist_dias_uci_previos             |          -0.0001 |
| dxp_neumonia                      |          -0.0001 |
| dxp_cancer                        |          -0.0002 |
| hist_ingresos_previos             |          -0.0012 |
| edad                              |          -0.0029 |

## Pasa de UCI a piso en 3 días o menos (salida temprana con certeza viva)

Tasa: 27%. Elegido: **ensamble_promedio** con variables **completo** — AUC CV 0.681, AUC prueba 0.683 (IC95 0.649-0.714). Corte ALTA probabilidad de salida temprana: p ≥ 0.31.

### Las 10 mejores combinaciones (por AUC CV)

| objetivo   | conjunto          | familia           |   auc_cv |   auc_prueba | ic95_prueba   |   ap_prueba |   brier_prueba |   intentos |
|:-----------|:------------------|:------------------|---------:|-------------:|:--------------|------------:|---------------:|-----------:|
| piso_3d    | completo          | ensamble_promedio |   0.681  |       0.6831 | 0.649-0.714   |      0.3884 |         0.1703 |          0 |
| piso_3d    | completo          | ensamble_apilado  |   0.6804 |       0.6833 | 0.649-0.713   |      0.3892 |         0.167  |          0 |
| piso_3d    | completo          | lightgbm          |   0.6801 |       0.6796 | 0.646-0.712   |      0.3845 |         0.166  |        144 |
| piso_3d    | completo          | xgboost           |   0.6793 |       0.6816 | 0.646-0.713   |      0.3845 |         0.1665 |         41 |
| piso_3d    | base+historia+dx3 | random_forest     |   0.6792 |       0.6846 | 0.649-0.716   |      0.3934 |         0.1654 |         21 |
| piso_3d    | base+historia+dx3 | catboost          |   0.6792 |       0.6861 | 0.650-0.719   |      0.3942 |         0.1655 |         17 |
| piso_3d    | completo          | random_forest     |   0.6791 |       0.6814 | 0.646-0.714   |      0.3918 |         0.2209 |         24 |
| piso_3d    | base+historia+dx3 | lightgbm          |   0.6786 |       0.6914 | 0.656-0.723   |      0.3969 |         0.1644 |        200 |
| piso_3d    | base+historia+dx3 | xgboost           |   0.6783 |       0.6864 | 0.652-0.719   |      0.391  |         0.1655 |         80 |
| piso_3d    | base+historia+dx3 | red_neuronal      |   0.6779 |       0.6837 | 0.647-0.716   |      0.3646 |         0.1684 |         80 |

### Mejor AUC CV por conjunto de variables

| conjunto          |   auc_cv |   auc_prueba |
|:------------------|---------:|-------------:|
| base              |   0.6775 |       0.685  |
| base+historia     |   0.6772 |       0.6843 |
| base+historia+dx3 |   0.6792 |       0.6914 |
| completo          |   0.681  |       0.684  |

### Importancia de variables (permutación, prueba)

|                                   |   pérdida de AUC |
|:----------------------------------|-----------------:|
| puntuacion_sofa_inicial           |           0.108  |
| dx3                               |           0.0254 |
| aseguradora                       |           0.0192 |
| capitulo_dx                       |           0.0057 |
| procedencia_piso                  |           0.0026 |
| dxp_obstetrico                    |           0.0007 |
| dxp_hta                           |           0.0006 |
| dxp_cardiopatia_isquemica_cronica |           0.0004 |
| dxp_cancer                        |           0.0004 |
| dxp_diabetes                      |           0.0003 |
| dxp_desnutricion                  |           0.0002 |
| dxp_erc                           |           0.0001 |
| dxp_arritmia                      |           0.0001 |
| dxp_trauma                        |           0      |
| dxp_falla_cardiaca                |           0      |
| dxp_hemorragia_digestiva          |           0      |
| hist_reingreso_30d                |           0      |
| dxp_valvulopatia                  |           0      |
| dxp_falla_respiratoria            |           0      |
| hist_ingresos_previos             |          -0      |
| dxp_neumonia                      |          -0      |
| dxp_sepsis                        |          -0.0001 |
| barthel_inicial                   |          -0.0003 |
| hist_dias_uci_previos             |          -0.0003 |
| dxp_iam                           |          -0.0004 |
| dxp_acv                           |          -0.0004 |
| edad                              |          -0.0028 |
| carga_uci_dia_ingreso             |          -0.0055 |
