### A. Default regime and ceiling

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_A_haw_baseline_test | 200 | 98.5% | 0.9 | 99.0% | 7.8 | 13.8 | {'success': 198, 'unsafe_site': 2} | 1 |
| P_A_haw_baseline_test_fixed | 200 | 98.5% | 0.9 | 99.0% | 7.8 | 13.8 | {'success': 198, 'unsafe_site': 2} | 1 |
| P_A_haw_baseline_train | 200 | 95.5% | 1.5 | 96.5% | 7.5 | 13.8 | {'success': 193, 'unsafe_site': 7} | 2 |
| P_A_haw_baseline_train_fixed | 200 | 95.5% | 1.5 | 96.5% | 7.6 | 13.8 | {'success': 193, 'unsafe_site': 7} | 2 |
| P_A_haw_oracle_test | 200 | 98.0% | 1.0 | 98.0% | 0.0 | 0.1 | {'success': 196, 'unsafe_site': 4} | 200 |
| P_A_haw_oracle_train | 200 | 97.5% | 1.1 | 97.5% | 0.0 | 0.1 | {'success': 195, 'unsafe_site': 5} | 200 |
| P_A_haw_wait_test | 200 | 98.0% | 1.0 | 98.0% | 7.9 | 13.7 | {'success': 196, 'unsafe_site': 3, 'timeout': 1} | 1 |
| P_A_haw_wait_test_fixed | 200 | 98.0% | 1.0 | 98.0% | 7.8 | 13.7 | {'success': 196, 'unsafe_site': 3, 'timeout': 1} | 1 |
| P_A_haw_wait_train | 200 | 96.5% | 1.3 | 97.0% | 7.8 | 13.3 | {'success': 194, 'unsafe_site': 6} | 1 |
| P_A_haw_wait_train_fixed | 200 | 96.5% | 1.3 | 97.0% | 7.8 | 13.3 | {'success': 194, 'unsafe_site': 6} | 1 |
| P_A_site04_baseline | 200 | 96.5% | 1.3 | 97.0% | 7.4 | 13.4 | {'success': 194, 'unsafe_site': 5, 'timeout': 1} | 2 |
| P_A_site04_baseline_fixed | 200 | 96.5% | 1.3 | 97.0% | 7.4 | 13.4 | {'success': 194, 'unsafe_site': 5, 'timeout': 1} | 2 |
| P_A_site04_hold | 200 | 96.5% | 1.3 | 96.5% | 7.4 | 14.2 | {'success': 193, 'unsafe_site': 6, 'hard_landing': 1} | 1 |
| P_A_site04_hold_fixed | 200 | 97.0% | 1.2 | 97.0% | 7.4 | 14.2 | {'success': 194, 'unsafe_site': 6} | 1 |
| P_A_site04_oracle | 200 | 98.5% | 0.9 | 98.5% | 0.0 | 0.1 | {'success': 197, 'unsafe_site': 2, 'timeout': 1} | 200 |
| P_A_site23_baseline | 200 | 99.0% | 0.7 | 99.0% | 8.4 | 15.2 | {'success': 198, 'unsafe_site': 2} | 0 |
| P_A_site23_baseline_fixed | 200 | 99.0% | 0.7 | 99.0% | 8.4 | 15.2 | {'success': 198, 'unsafe_site': 2} | 0 |
| P_A_site23_hold | 200 | 98.0% | 1.0 | 98.0% | 8.8 | 15.0 | {'success': 196, 'unsafe_site': 3, 'timeout': 1} | 0 |
| P_A_site23_hold_fixed | 200 | 98.5% | 0.9 | 98.5% | 8.9 | 15.0 | {'success': 197, 'unsafe_site': 3} | 0 |
| P_A_site23_oracle | 200 | 97.5% | 1.1 | 97.5% | 0.0 | 0.0 | {'success': 195, 'unsafe_site': 5} | 200 |

### B. Initial-error sweep, realistic sensor/map (narrow / wide / +alt_sync / +fov 60 / hold)

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_B_site04_1000_narrow | 100 | 12.0% | 3.2 | 19.0% | 14.8 | 933.2 | {'timeout': 52, 'hard_landing': 10, 'unsafe_site': 15, 'success': 19, 'out_of_bounds': 4} | 0 |
| P_B_site04_1000_sync | 100 | 99.0% | 1.0 | 99.0% | 8.2 | 13.8 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site04_1000_sync_fov60 | 100 | 98.0% | 1.4 | 98.0% | 7.7 | 13.6 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_B_site04_1000_sync_hold | 100 | 98.0% | 1.4 | 98.0% | 7.9 | 13.8 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_B_site04_1000_wide | 100 | 99.0% | 1.0 | 99.0% | 8.2 | 13.8 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site04_300_narrow | 100 | 81.0% | 3.9 | 90.0% | 9.1 | 41.1 | {'success': 90, 'timeout': 6, 'unsafe_site': 3, 'hard_landing': 1} | 0 |
| P_B_site04_300_sync | 100 | 98.0% | 1.4 | 98.0% | 8.0 | 13.8 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_B_site04_300_sync_fov60 | 100 | 98.0% | 1.4 | 98.0% | 8.1 | 13.2 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_B_site04_300_sync_hold | 100 | 98.0% | 1.4 | 98.0% | 7.7 | 14.2 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_B_site04_300_wide | 100 | 97.0% | 1.7 | 97.0% | 8.0 | 13.8 | {'success': 97, 'unsafe_site': 2, 'hard_landing': 1} | 0 |
| P_B_site04_600_narrow | 100 | 24.0% | 4.3 | 46.0% | 15.3 | 334.6 | {'success': 46, 'timeout': 28, 'hard_landing': 12, 'unsafe_site': 13, 'out_of_bounds': 1} | 0 |
| P_B_site04_600_sync | 100 | 99.0% | 1.0 | 99.0% | 8.1 | 13.9 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site04_600_sync_fov60 | 100 | 99.0% | 1.0 | 99.0% | 8.1 | 13.5 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site04_600_sync_hold | 100 | 98.0% | 1.4 | 98.0% | 7.9 | 14.0 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_B_site04_600_wide | 100 | 99.0% | 1.0 | 99.0% | 8.1 | 13.9 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site23_1000_narrow | 100 | 12.0% | 3.2 | 29.0% | 152.9 | 955.5 | {'unsafe_site': 21, 'hard_landing': 24, 'timeout': 26, 'success': 29} | 0 |
| P_B_site23_1000_sync | 100 | 96.0% | 2.0 | 96.0% | 8.3 | 14.3 | {'success': 96, 'unsafe_site': 2, 'hard_landing': 1, 'timeout': 1} | 3 |
| P_B_site23_1000_sync_fov60 | 100 | 97.0% | 1.7 | 97.0% | 8.6 | 14.7 | {'success': 97, 'unsafe_site': 2, 'hard_landing': 1} | 2 |
| P_B_site23_1000_sync_hold | 100 | 96.0% | 2.0 | 96.0% | 8.6 | 13.4 | {'success': 96, 'unsafe_site': 3, 'timeout': 1} | 3 |
| P_B_site23_1000_wide | 100 | 94.0% | 2.4 | 94.0% | 8.3 | 14.3 | {'success': 94, 'unsafe_site': 2, 'hard_landing': 2, 'timeout': 2} | 3 |
| P_B_site23_300_narrow | 100 | 81.0% | 3.9 | 91.0% | 9.3 | 70.3 | {'success': 91, 'unsafe_site': 4, 'timeout': 2, 'hard_landing': 3} | 0 |
| P_B_site23_300_sync | 100 | 99.0% | 1.0 | 99.0% | 8.4 | 14.0 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site23_300_sync_fov60 | 100 | 99.0% | 1.0 | 99.0% | 8.3 | 14.1 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site23_300_sync_hold | 100 | 99.0% | 1.0 | 99.0% | 9.1 | 13.8 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site23_300_wide | 100 | 99.0% | 1.0 | 99.0% | 8.4 | 14.0 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_B_site23_600_narrow | 100 | 25.0% | 4.3 | 54.0% | 65.7 | 346.8 | {'success': 54, 'timeout': 12, 'unsafe_site': 23, 'hard_landing': 11} | 0 |
| P_B_site23_600_sync | 100 | 90.0% | 3.0 | 91.0% | 8.8 | 14.0 | {'success': 91, 'unsafe_site': 9} | 9 |
| P_B_site23_600_sync_fov60 | 100 | 92.0% | 2.7 | 93.0% | 8.8 | 14.9 | {'success': 93, 'unsafe_site': 7} | 7 |
| P_B_site23_600_sync_hold | 100 | 91.0% | 2.9 | 92.0% | 9.1 | 13.9 | {'success': 92, 'unsafe_site': 8} | 8 |
| P_B_site23_600_wide | 100 | 89.0% | 3.1 | 90.0% | 8.6 | 14.1 | {'success': 90, 'unsafe_site': 9, 'hard_landing': 1} | 9 |

### D. Residual PPO

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_D_dec_baseline | 200 | 83.0% | 2.7 | 99.0% | 1.5 | 110.4 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 32 |
| P_D_dec_cov_s0 | 200 | 59.0% | 3.5 | 74.5% | 10.5 | 160.8 | {'success': 149, 'hard_landing': 50, 'unsafe_site': 1} | 41 |
| P_D_dec_cov_s1 | 200 | 66.5% | 3.3 | 82.0% | 7.0 | 133.0 | {'hard_landing': 35, 'success': 164, 'unsafe_site': 1} | 40 |
| P_D_dec_cov_s2 | 200 | 79.5% | 2.9 | 97.5% | 6.2 | 135.5 | {'success': 195, 'hard_landing': 4, 'unsafe_site': 1} | 41 |
| P_D_dec_hold | 200 | 89.0% | 2.2 | 99.5% | 1.7 | 60.3 | {'success': 199, 'unsafe_site': 1} | 21 |
| P_D_dec_nocov_s0 | 200 | 82.5% | 2.7 | 98.5% | 2.9 | 104.2 | {'success': 197, 'unsafe_site': 2, 'hard_landing': 1} | 34 |
| P_D_dec_nocov_s1 | 200 | 77.0% | 3.0 | 97.0% | 5.1 | 140.0 | {'success': 194, 'hard_landing': 4, 'unsafe_site': 2} | 43 |
| P_D_dec_nocov_s2 | 200 | 79.0% | 2.9 | 99.0% | 3.5 | 139.3 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 40 |
| P_D_haw_nocov_s1_best | 200 | 98.5% | 0.9 | 99.5% | 5.7 | 8.3 | {'success': 199, 'unsafe_site': 1} | 2 |
| P_D_haw_nocov_s1_final | 200 | 87.5% | 2.3 | 88.0% | 9.4 | 13.4 | {'success': 176, 'hard_landing': 21, 'unsafe_site': 3} | 2 |
| P_D_haw_nocov_s2_best | 200 | 99.5% | 0.5 | 100.0% | 1.7 | 3.3 | {'success': 200} | 1 |
| P_D_haw_nocov_s2_final | 200 | 98.0% | 1.0 | 98.5% | 5.1 | 7.6 | {'success': 197, 'timeout': 3} | 1 |

### E. Learned hold/detour (nominal conditions)

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_E_baseline | 200 | 83.0% | 2.7 | 99.0% | 1.5 | 110.4 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 32 |
| P_E_hold | 200 | 89.0% | 2.2 | 99.5% | 1.7 | 60.3 | {'success': 199, 'unsafe_site': 1} | 21 |
| P_E_holdall | 200 | 87.5% | 2.3 | 99.0% | 1.7 | 74.1 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 22 |
| P_E_learned_s0 | 200 | 87.5% | 2.3 | 99.0% | 1.7 | 74.1 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 22 |
| P_E_learned_s1 | 200 | 87.5% | 2.3 | 99.0% | 1.7 | 74.1 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 22 |
| P_E_learned_s2 | 200 | 87.5% | 2.3 | 99.0% | 1.7 | 74.1 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 22 |
| P_E_oracle | 200 | 99.5% | 0.5 | 99.5% | 0.0 | 0.1 | {'success': 199, 'unsafe_site': 1} | 200 |
| P_E_rule | 200 | 75.5% | 3.0 | 77.5% | 1.4 | 3.9 | {'success': 155, 'timeout': 43, 'unsafe_site': 1, 'hard_landing': 1} | 9 |

### H. Rule restricted to the first decision

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_H_rule_first_nominal | 200 | 90.0% | 2.1 | 99.0% | 1.5 | 18.5 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 17 |

### L. Legacy idealised-map conditions

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_L_s23_120_legacy | 100 | 95.0% | 2.2 | 96.0% | 1.2 | 2.6 | {'success': 96, 'unsafe_site': 4} | 0 |
| P_L_s23_120_sync31 | 100 | 96.0% | 2.0 | 96.0% | 0.8 | 1.5 | {'success': 96, 'unsafe_site': 4} | 0 |
| P_L_s23_600_legacy | 100 | 73.0% | 4.4 | 79.0% | 1.2 | 3.9 | {'success': 79, 'timeout': 3, 'unsafe_site': 15, 'hard_landing': 3} | 17 |
| P_L_s23_600_sync31 | 100 | 88.0% | 3.2 | 89.0% | 0.8 | 1.7 | {'success': 89, 'unsafe_site': 11} | 9 |

### R. Realism sensitivity and RL robustness

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_R_dec_baseline | 200 | 73.0% | 3.1 | 100.0% | 10.7 | 167.0 | {'success': 200} | 49 |
| P_R_dec_hold | 200 | 78.0% | 2.9 | 100.0% | 10.3 | 148.8 | {'success': 200} | 36 |
| P_R_dec_learned_s0 | 200 | 77.5% | 3.0 | 100.0% | 10.0 | 150.6 | {'success': 200} | 42 |
| P_R_dec_learned_s1 | 200 | 77.5% | 3.0 | 100.0% | 10.0 | 150.6 | {'success': 200} | 42 |
| P_R_dec_learned_s2 | 200 | 77.5% | 3.0 | 100.0% | 10.0 | 150.6 | {'success': 200} | 42 |
| P_R_dec_rule | 200 | 76.5% | 3.0 | 84.5% | 8.6 | 37.1 | {'success': 169, 'timeout': 31} | 21 |
| P_R_fov20_s120 | 100 | 97.0% | 1.7 | 97.0% | 8.2 | 13.5 | {'success': 97, 'unsafe_site': 3} | 0 |
| P_R_fov20_s600 | 100 | 81.0% | 3.9 | 85.0% | 8.8 | 15.4 | {'success': 85, 'unsafe_site': 15} | 17 |
| P_R_fov30_s120 | 100 | 98.0% | 1.4 | 98.0% | 8.3 | 13.1 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_R_fov30_s600 | 100 | 86.0% | 3.5 | 87.0% | 8.8 | 13.8 | {'success': 87, 'unsafe_site': 13} | 12 |
| P_R_fov60_s120 | 100 | 99.0% | 1.0 | 99.0% | 8.5 | 13.9 | {'success': 99, 'unsafe_site': 1} | 0 |
| P_R_fov60_s600 | 100 | 92.0% | 2.7 | 93.0% | 8.8 | 14.9 | {'success': 93, 'unsafe_site': 7} | 7 |
| P_R_noise0_s120 | 100 | 96.0% | 2.0 | 96.0% | 7.6 | 12.6 | {'success': 96, 'unsafe_site': 4} | 0 |
| P_R_noise0_s600 | 100 | 89.0% | 3.1 | 90.0% | 8.0 | 13.0 | {'success': 90, 'unsafe_site': 10} | 8 |
| P_R_noise2_s120 | 100 | 97.0% | 1.7 | 98.0% | 10.4 | 18.1 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_R_noise2_s600 | 100 | 89.0% | 3.1 | 91.0% | 9.8 | 18.8 | {'success': 91, 'unsafe_site': 9} | 7 |
| P_R_noise3_s120 | 100 | 97.0% | 1.7 | 98.0% | 13.3 | 23.0 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_R_noise3_s600 | 100 | 88.0% | 3.2 | 90.0% | 13.8 | 25.1 | {'success': 90, 'unsafe_site': 8, 'timeout': 2} | 6 |
| P_R_off0_s120 | 100 | 98.0% | 1.4 | 98.0% | 3.8 | 6.7 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_R_off0_s600 | 100 | 90.0% | 3.0 | 91.0% | 3.8 | 6.8 | {'success': 91, 'unsafe_site': 9} | 9 |
| P_R_off12_s120 | 100 | 98.0% | 1.4 | 98.0% | 14.8 | 24.9 | {'success': 98, 'unsafe_site': 2} | 0 |
| P_R_off12_s600 | 100 | 89.0% | 3.1 | 90.0% | 15.1 | 26.2 | {'success': 90, 'unsafe_site': 10} | 9 |
| P_R_off3_s120 | 100 | 100.0% | 0.0 | 100.0% | 5.5 | 9.4 | {'success': 100} | 0 |
| P_R_off3_s600 | 100 | 91.0% | 2.9 | 92.0% | 5.7 | 9.6 | {'success': 92, 'unsafe_site': 8} | 9 |

### S. Extra training seeds, residual scale k=1.0, cold-start decision policy, away rules

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_S_dec_away | 200 | 73.5% | 3.1 | 76.0% | 1.4 | 4.6 | {'success': 152, 'timeout': 45, 'unsafe_site': 1, 'out_of_bounds': 1, 'hard_landing': 1} | 16 |
| P_S_dec_away400 | 200 | 86.5% | 2.4 | 89.0% | 1.5 | 5.0 | {'success': 178, 'timeout': 20, 'unsafe_site': 1, 'hard_landing': 1} | 5 |
| P_S_dec_cold_s0 | 200 | 81.5% | 2.7 | 99.0% | 1.9 | 134.8 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 35 |
| P_S_dec_cold_s1 | 200 | 86.0% | 2.5 | 97.0% | 1.7 | 75.3 | {'success': 194, 'timeout': 4, 'unsafe_site': 1, 'hard_landing': 1} | 22 |
| P_S_dec_cold_s2 | 200 | 84.0% | 2.6 | 97.5% | 1.6 | 97.3 | {'success': 195, 'timeout': 3, 'unsafe_site': 1, 'hard_landing': 1} | 26 |
| P_S_dec_cold_s3 | 200 | 81.5% | 2.7 | 99.0% | 1.9 | 134.8 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 35 |
| P_S_dec_cold_s4 | 200 | 81.5% | 2.7 | 99.0% | 1.9 | 134.8 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 35 |
| P_S_dec_cov_s3 | 200 | 78.5% | 2.9 | 98.0% | 8.9 | 154.1 | {'success': 196, 'unsafe_site': 2, 'hard_landing': 2} | 42 |
| P_S_dec_cov_s4 | 200 | 78.0% | 2.9 | 98.0% | 4.5 | 144.4 | {'success': 196, 'unsafe_site': 2, 'hard_landing': 2} | 43 |
| P_S_dec_k1_nocov_s0 | 200 | 82.0% | 2.7 | 98.5% | 4.2 | 132.0 | {'success': 197, 'unsafe_site': 2, 'hard_landing': 1} | 34 |
| P_S_dec_k1_nocov_s1 | 200 | 77.0% | 3.0 | 95.0% | 9.7 | 145.9 | {'success': 190, 'hard_landing': 8, 'unsafe_site': 2} | 40 |
| P_S_dec_k1_nocov_s2 | 200 | 70.5% | 3.2 | 84.0% | 13.1 | 136.8 | {'success': 168, 'hard_landing': 30, 'unsafe_site': 2} | 35 |
| P_S_dec_nocov_s3 | 200 | 72.5% | 3.2 | 90.5% | 4.7 | 138.7 | {'success': 181, 'hard_landing': 17, 'unsafe_site': 2} | 41 |
| P_S_dec_nocov_s4 | 200 | 76.0% | 3.0 | 93.0% | 10.8 | 150.0 | {'success': 186, 'hard_landing': 12, 'unsafe_site': 2} | 42 |
| P_S_real_away | 200 | 77.5% | 3.0 | 82.0% | 8.7 | 22.0 | {'success': 164, 'timeout': 35, 'out_of_bounds': 1} | 17 |
| P_S_real_away400 | 200 | 86.5% | 2.4 | 93.5% | 9.3 | 28.8 | {'success': 187, 'timeout': 13} | 13 |
| P_S_real_cold_s0 | 200 | 73.5% | 3.1 | 100.0% | 10.9 | 164.3 | {'success': 200} | 49 |
| P_S_real_cold_s1 | 200 | 78.5% | 2.9 | 99.0% | 10.7 | 144.5 | {'success': 198, 'timeout': 2} | 33 |
| P_S_real_cold_s2 | 200 | 78.0% | 2.9 | 99.5% | 9.9 | 140.1 | {'success': 199, 'timeout': 1} | 40 |
| P_S_real_cold_s3 | 200 | 73.5% | 3.1 | 100.0% | 10.9 | 164.3 | {'success': 200} | 49 |
| P_S_real_cold_s4 | 200 | 73.5% | 3.1 | 100.0% | 10.9 | 164.3 | {'success': 200} | 49 |

### S2. Away rules at the first decision only; cold-start policies sampled

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_S2_dec_away400first | 200 | 91.0% | 2.0 | 99.0% | 1.8 | 11.6 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 12 |
| P_S2_dec_awayfirst | 200 | 93.5% | 1.7 | 99.0% | 1.5 | 6.4 | {'success': 198, 'unsafe_site': 1, 'hard_landing': 1} | 8 |
| P_S2_dec_sto_s0 | 200 | 87.0% | 2.4 | 95.5% | 1.5 | 7.0 | {'success': 191, 'timeout': 7, 'unsafe_site': 1, 'hard_landing': 1} | 14 |
| P_S2_dec_sto_s1 | 200 | 93.0% | 1.8 | 97.5% | 1.6 | 6.9 | {'success': 195, 'unsafe_site': 1, 'timeout': 3, 'hard_landing': 1} | 8 |
| P_S2_dec_sto_s2 | 200 | 90.0% | 2.1 | 98.5% | 1.6 | 16.0 | {'success': 197, 'unsafe_site': 1, 'timeout': 1, 'hard_landing': 1} | 14 |
| P_S2_dec_sto_s3 | 200 | 87.5% | 2.3 | 97.5% | 1.7 | 48.2 | {'success': 195, 'unsafe_site': 1, 'timeout': 3, 'hard_landing': 1} | 19 |
| P_S2_dec_sto_s4 | 200 | 90.0% | 2.1 | 98.5% | 1.6 | 14.8 | {'success': 197, 'unsafe_site': 1, 'timeout': 1, 'hard_landing': 1} | 14 |
| P_S2_real_away400first | 200 | 81.5% | 2.7 | 100.0% | 10.0 | 131.9 | {'success': 200} | 32 |
| P_S2_real_awayfirst | 200 | 84.5% | 2.6 | 99.5% | 9.8 | 97.3 | {'success': 199, 'timeout': 1} | 21 |
| P_S2_real_sto_s0 | 200 | 77.0% | 3.0 | 97.5% | 10.3 | 138.6 | {'success': 195, 'timeout': 5} | 32 |
| P_S2_real_sto_s1 | 200 | 78.0% | 2.9 | 98.0% | 10.2 | 180.7 | {'success': 196, 'timeout': 4} | 32 |
| P_S2_real_sto_s2 | 200 | 81.0% | 2.8 | 100.0% | 10.2 | 135.8 | {'success': 200} | 30 |
| P_S2_real_sto_s3 | 200 | 83.0% | 2.7 | 100.0% | 9.8 | 131.4 | {'success': 200} | 29 |
| P_S2_real_sto_s4 | 200 | 79.5% | 2.9 | 99.0% | 9.7 | 137.9 | {'success': 198, 'timeout': 2} | 34 |

### X. Extended Site 23 runs (400 episodes, sigma 600 / 1000)

| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |
|---|---|---|---|---|---|---|---|---|
| P_X_site23_1000_sync | 400 | 92.2% | 1.3 | 93.0% | 8.2 | 15.3 | {'success': 372, 'unsafe_site': 17, 'hard_landing': 4, 'timeout': 5, 'out_of_bounds': 2} | 16 |
| P_X_site23_1000_sync_fov60 | 400 | 95.5% | 1.0 | 96.0% | 8.3 | 15.6 | {'success': 384, 'unsafe_site': 11, 'hard_landing': 2, 'out_of_bounds': 2, 'timeout': 1} | 6 |
| P_X_site23_1000_wide | 400 | 90.0% | 1.5 | 90.8% | 8.2 | 15.1 | {'success': 363, 'unsafe_site': 16, 'hard_landing': 15, 'timeout': 4, 'out_of_bounds': 2} | 16 |
| P_X_site23_600_sync | 400 | 88.0% | 1.6 | 90.0% | 8.3 | 15.3 | {'success': 360, 'unsafe_site': 39, 'hard_landing': 1} | 38 |
| P_X_site23_600_sync_fov20 | 400 | 76.8% | 2.1 | 81.5% | 8.1 | 16.1 | {'success': 326, 'unsafe_site': 67, 'hard_landing': 4, 'timeout': 3} | 75 |
| P_X_site23_600_sync_fov30 | 400 | 85.5% | 1.8 | 88.2% | 8.2 | 15.3 | {'success': 353, 'unsafe_site': 46, 'hard_landing': 1} | 48 |
| P_X_site23_600_sync_fov60 | 400 | 89.2% | 1.5 | 91.2% | 8.2 | 15.9 | {'success': 365, 'unsafe_site': 34, 'hard_landing': 1} | 32 |
| P_X_site23_600_sync_hold | 400 | 88.0% | 1.6 | 90.0% | 8.3 | 15.1 | {'success': 360, 'unsafe_site': 35, 'timeout': 4, 'hard_landing': 1} | 38 |
| P_X_site23_600_wide | 400 | 86.0% | 1.7 | 88.0% | 8.1 | 15.2 | {'success': 352, 'unsafe_site': 38, 'hard_landing': 9, 'timeout': 1} | 38 |

### C. Failure diagnosis (category counts per run)

| run | n | categories |
|---|---|---|
| P_C_diag_1000_legacy | 60 | {'precise': 57, 'nav_ok_hazard': 1, 'nav_bad': 1, 'other': 1} |
| P_C_diag_1000_sync | 60 | {'precise': 59, 'nav_ok_hazard': 1} |
| P_C_diag_1000_sync_fov60 | 60 | {'precise': 59, 'nav_ok_hazard': 1} |
| P_C_diag_600_legacy | 60 | {'precise': 53, 'zero_fix': 5, 'nav_ok_hazard': 2} |
| P_C_diag_600_sync | 60 | {'precise': 54, 'zero_fix': 5, 'nav_ok_hazard': 1} |
| P_C_diag_600_sync_fov60 | 60 | {'precise': 54, 'zero_fix': 5, 'nav_ok_hazard': 1} |

### H. Decision-layer headroom (first-decision action tried on every episode; precise rate)

| run | N | episodes with a decision point | no detour | hover | mean single detour | best single action | hindsight best of all |
|---|---|---|---|---|---|---|---|
| P_H_test_nominal | 200 | 110 | 81.5% | 86.0% | 89.8% | 92.5% | 98.5% |
| P_H_test_realistic | 200 | 88 | 73.5% | 76.0% | 79.3% | 83.5% | 99.5% |
| P_H_train_nominal | 400 | 217 | 84.5% | 86.0% | 89.8% | 93.5% | 99.5% |
| P_H_train_realistic | 400 | 165 | 73.0% | 70.8% | 77.2% | 80.5% | 95.8% |

### H2. Learned first-decision chooser (offline fit on train logs, exact evaluation on test logs)

| run | test N | none | hover | best fixed | random detour | matchability rule | away-800 rule | away-400 rule | conditional logit | hindsight best |
|---|---|---|---|---|---|---|---|---|---|---|
| P_H_fit_nominal | 200 | 81.5% | 86.0% | 89.5% | 89.7% | 90.0% | 93.5% | 91.0% | 94.5% | 98.5% |
| P_H_fit_realistic | 200 | 73.5% | 76.0% | 81.0% | 79.3% | 81.5% | 84.5% | 81.5% | 83.5% | 99.5% |

### Paired comparisons (same seeds; 95% paired-bootstrap CI; run scripts/paired_ci.py first)

| comparison (A minus B) | n | A | B | diff [pts] | 95% CI | only A / only B | sign-test p |
|---|---|---|---|---|---|---|---|
| A site04: baseline vs oracle | 200 | 96.5% | 98.5% | -2.0 | -4.5 to +0.0 | 1 / 5 | 0.219 |
| A site04: alt_sync vs legacy windows (baseline) | 200 | 96.5% | 96.5% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| A site04: hold vs baseline (alt_sync) | 200 | 97.0% | 96.5% | +0.5 | -1.0 to +2.0 | 2 / 1 | 1.000 |
| A site23: baseline vs oracle | 200 | 99.0% | 97.5% | +1.5 | -1.0 to +4.0 | 5 / 2 | 0.453 |
| A site23: alt_sync vs legacy windows (baseline) | 200 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| A site23: hold vs baseline (alt_sync) | 200 | 98.5% | 99.0% | -0.5 | -1.5 to +0.0 | 0 / 1 | 1.000 |
| A haworth test: baseline vs oracle | 200 | 98.5% | 98.0% | +0.5 | -1.0 to +2.5 | 2 / 1 | 1.000 |
| A haworth test: hold(wait70) vs baseline (alt_sync) | 200 | 98.0% | 98.5% | -0.5 | -1.5 to +0.0 | 0 / 1 | 1.000 |
| A haworth test: alt_sync vs legacy (baseline) | 200 | 98.5% | 98.5% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| A haworth train: baseline vs oracle | 200 | 95.5% | 97.5% | -2.0 | -4.5 to +0.0 | 1 / 5 | 0.219 |
| A haworth train: hold(wait70) vs baseline (alt_sync) | 200 | 96.5% | 95.5% | +1.0 | +0.0 to +2.5 | 2 / 0 | 0.500 |
| A haworth train: alt_sync vs legacy (baseline) | 200 | 95.5% | 95.5% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site04 sigma300: wide vs narrow window | 100 | 97.0% | 81.0% | +16.0 | +9.0 to +23.0 | 16 / 0 | 0.000 |
| B site04 sigma300: alt_sync vs wide | 100 | 98.0% | 97.0% | +1.0 | +0.0 to +3.0 | 1 / 0 | 1.000 |
| B site04 sigma300: sync+fov60 vs sync | 100 | 98.0% | 98.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site04 sigma300: hold vs baseline (sync) | 100 | 98.0% | 98.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site04 sigma600: wide vs narrow window | 100 | 99.0% | 24.0% | +75.0 | +66.0 to +83.0 | 75 / 0 | 0.000 |
| B site04 sigma600: alt_sync vs wide | 100 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site04 sigma600: sync+fov60 vs sync | 100 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site04 sigma600: hold vs baseline (sync) | 100 | 98.0% | 99.0% | -1.0 | -3.0 to +0.0 | 0 / 1 | 1.000 |
| B site04 sigma1000: wide vs narrow window | 100 | 99.0% | 12.0% | +87.0 | +80.0 to +93.0 | 87 / 0 | 0.000 |
| B site04 sigma1000: alt_sync vs wide | 100 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site04 sigma1000: sync+fov60 vs sync | 100 | 98.0% | 99.0% | -1.0 | -3.0 to +0.0 | 0 / 1 | 1.000 |
| B site04 sigma1000: hold vs baseline (sync) | 100 | 98.0% | 99.0% | -1.0 | -3.0 to +0.0 | 0 / 1 | 1.000 |
| B site23 sigma300: wide vs narrow window | 100 | 99.0% | 81.0% | +18.0 | +11.0 to +26.0 | 18 / 0 | 0.000 |
| B site23 sigma300: alt_sync vs wide | 100 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site23 sigma300: sync+fov60 vs sync | 100 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site23 sigma300: hold vs baseline (sync) | 100 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| B site23 sigma600: wide vs narrow window | 100 | 89.0% | 25.0% | +64.0 | +53.0 to +74.0 | 66 / 2 | 0.000 |
| B site23 sigma600: alt_sync vs wide | 100 | 90.0% | 89.0% | +1.0 | +0.0 to +3.0 | 1 / 0 | 1.000 |
| B site23 sigma600: sync+fov60 vs sync | 100 | 92.0% | 90.0% | +2.0 | +0.0 to +5.0 | 2 / 0 | 0.500 |
| B site23 sigma600: hold vs baseline (sync) | 100 | 91.0% | 90.0% | +1.0 | +0.0 to +3.0 | 1 / 0 | 1.000 |
| B site23 sigma1000: wide vs narrow window | 100 | 94.0% | 12.0% | +82.0 | +74.0 to +89.0 | 82 / 0 | 0.000 |
| B site23 sigma1000: alt_sync vs wide | 100 | 96.0% | 94.0% | +2.0 | +0.0 to +5.0 | 2 / 0 | 0.500 |
| B site23 sigma1000: sync+fov60 vs sync | 100 | 97.0% | 96.0% | +1.0 | +0.0 to +3.0 | 1 / 0 | 1.000 |
| B site23 sigma1000: hold vs baseline (sync) | 100 | 96.0% | 96.0% | +0.0 | -3.0 to +3.0 | 1 / 1 | 1.000 |
| D synthetic: PPO nocov s0 vs baseline | 200 | 82.5% | 83.0% | -0.5 | -4.5 to +3.0 | 7 / 8 | 1.000 |
| D synthetic: PPO nocov s1 vs baseline | 200 | 77.0% | 83.0% | -6.0 | -11.0 to -1.5 | 5 / 17 | 0.017 |
| D synthetic: PPO nocov s2 vs baseline | 200 | 79.0% | 83.0% | -4.0 | -8.5 to +0.5 | 7 / 15 | 0.134 |
| D synthetic: PPO cov s0 vs baseline | 200 | 59.0% | 83.0% | -24.0 | -31.5 to -16.5 | 8 / 56 | 0.000 |
| D synthetic: PPO cov s1 vs baseline | 200 | 66.5% | 83.0% | -16.5 | -23.0 to -10.0 | 9 / 42 | 0.000 |
| D synthetic: PPO cov s2 vs baseline | 200 | 79.5% | 83.0% | -3.5 | -8.0 to +0.5 | 6 / 13 | 0.167 |
| D synthetic: hold vs baseline | 200 | 89.0% | 83.0% | +6.0 | +2.0 to +10.0 | 15 / 3 | 0.008 |
| D haworth test: PPO nocov s1 best vs baseline | 200 | 98.5% | 98.5% | +0.0 | -2.0 to +2.0 | 2 / 2 | 1.000 |
| D haworth test: PPO nocov s1 final vs baseline | 200 | 87.5% | 98.5% | -11.0 | -15.5 to -6.5 | 1 / 23 | 0.000 |
| D haworth test: PPO nocov s2 best vs baseline | 200 | 99.5% | 98.5% | +1.0 | +0.0 to +2.5 | 2 / 0 | 0.500 |
| D haworth test: PPO nocov s2 final vs baseline | 200 | 98.0% | 98.5% | -0.5 | -2.5 to +1.5 | 2 / 3 | 1.000 |
| E: learned s0 vs baseline | 200 | 87.5% | 83.0% | +4.5 | +1.0 to +8.5 | 12 / 3 | 0.035 |
| E: learned s0 vs hold | 200 | 87.5% | 89.0% | -1.5 | -4.0 to +1.0 | 2 / 5 | 0.453 |
| R decision: learned s0 vs baseline | 200 | 77.5% | 73.0% | +4.5 | +1.5 to +8.0 | 10 / 1 | 0.012 |
| R decision: learned s0 vs hold | 200 | 77.5% | 78.0% | -0.5 | -4.5 to +3.5 | 8 / 9 | 1.000 |
| E: learned s1 vs baseline | 200 | 87.5% | 83.0% | +4.5 | +1.0 to +8.5 | 12 / 3 | 0.035 |
| E: learned s1 vs hold | 200 | 87.5% | 89.0% | -1.5 | -4.0 to +1.0 | 2 / 5 | 0.453 |
| R decision: learned s1 vs baseline | 200 | 77.5% | 73.0% | +4.5 | +1.5 to +8.0 | 10 / 1 | 0.012 |
| R decision: learned s1 vs hold | 200 | 77.5% | 78.0% | -0.5 | -4.5 to +3.5 | 8 / 9 | 1.000 |
| E: learned s2 vs baseline | 200 | 87.5% | 83.0% | +4.5 | +1.0 to +8.5 | 12 / 3 | 0.035 |
| E: learned s2 vs hold | 200 | 87.5% | 89.0% | -1.5 | -4.0 to +1.0 | 2 / 5 | 0.453 |
| R decision: learned s2 vs baseline | 200 | 77.5% | 73.0% | +4.5 | +1.5 to +8.0 | 10 / 1 | 0.012 |
| R decision: learned s2 vs hold | 200 | 77.5% | 78.0% | -0.5 | -4.5 to +3.5 | 8 / 9 | 1.000 |
| E: hold vs baseline | 200 | 89.0% | 83.0% | +6.0 | +2.0 to +10.0 | 15 / 3 | 0.008 |
| E: rule vs baseline | 200 | 75.5% | 83.0% | -7.5 | -14.0 to -1.5 | 14 / 29 | 0.032 |
| E: baseline vs oracle | 200 | 83.0% | 99.5% | -16.5 | -22.0 to -11.5 | 0 / 33 | 0.000 |
| R decision: hold vs baseline | 200 | 78.0% | 73.0% | +5.0 | +0.5 to +9.5 | 17 / 7 | 0.064 |
| R decision: rule vs baseline | 200 | 76.5% | 73.0% | +3.5 | -2.5 to +9.5 | 22 / 15 | 0.324 |
| R site23 sigma120: fov20 vs preset | 100 | 97.0% | 99.0% | -2.0 | -5.0 to +0.0 | 0 / 2 | 0.500 |
| R site23 sigma600: fov20 vs preset | 100 | 81.0% | 90.0% | -9.0 | -15.0 to -4.0 | 0 / 9 | 0.004 |
| R site23 sigma120: fov30 vs preset | 100 | 98.0% | 99.0% | -1.0 | -3.0 to +0.0 | 0 / 1 | 1.000 |
| R site23 sigma600: fov30 vs preset | 100 | 86.0% | 90.0% | -4.0 | -8.0 to -1.0 | 0 / 4 | 0.125 |
| R site23 sigma120: fov60 vs preset | 100 | 99.0% | 99.0% | +0.0 | +0.0 to +0.0 | 0 / 0 | 1.000 |
| R site23 sigma600: fov60 vs preset | 100 | 92.0% | 90.0% | +2.0 | +0.0 to +5.0 | 2 / 0 | 0.500 |
| R site23 sigma120: off0 vs preset | 100 | 98.0% | 99.0% | -1.0 | -4.0 to +2.0 | 1 / 2 | 1.000 |
| R site23 sigma600: off0 vs preset | 100 | 90.0% | 90.0% | +0.0 | -3.0 to +3.0 | 1 / 1 | 1.000 |
| R site23 sigma120: off3 vs preset | 100 | 100.0% | 99.0% | +1.0 | +0.0 to +3.0 | 1 / 0 | 1.000 |
| R site23 sigma600: off3 vs preset | 100 | 91.0% | 90.0% | +1.0 | +0.0 to +3.0 | 1 / 0 | 1.000 |
| R site23 sigma120: off12 vs preset | 100 | 98.0% | 99.0% | -1.0 | -3.0 to +0.0 | 0 / 1 | 1.000 |
| R site23 sigma600: off12 vs preset | 100 | 89.0% | 90.0% | -1.0 | -3.0 to +0.0 | 0 / 1 | 1.000 |
| R site23 sigma120: noise0 vs preset | 100 | 96.0% | 99.0% | -3.0 | -8.0 to +1.0 | 1 / 4 | 0.375 |
| R site23 sigma600: noise0 vs preset | 100 | 89.0% | 90.0% | -1.0 | -5.0 to +3.0 | 2 / 3 | 1.000 |
| R site23 sigma120: noise2 vs preset | 100 | 97.0% | 99.0% | -2.0 | -6.0 to +2.0 | 1 / 3 | 0.625 |
| R site23 sigma600: noise2 vs preset | 100 | 89.0% | 90.0% | -1.0 | -6.0 to +4.0 | 3 / 4 | 1.000 |
| R site23 sigma120: noise3 vs preset | 100 | 97.0% | 99.0% | -2.0 | -6.0 to +2.0 | 1 / 3 | 0.625 |
| R site23 sigma600: noise3 vs preset | 100 | 88.0% | 90.0% | -2.0 | -9.0 to +5.0 | 5 / 7 | 0.774 |
| R site23 sigma120: fov60 vs fov20 | 100 | 99.0% | 97.0% | +2.0 | +0.0 to +5.0 | 2 / 0 | 0.500 |
| R site23 sigma600: fov60 vs fov20 | 100 | 92.0% | 81.0% | +11.0 | +5.0 to +17.0 | 11 / 0 | 0.001 |
| X site23 sigma600: alt_sync vs wide | 400 | 88.0% | 86.0% | +2.0 | +0.8 to +3.5 | 8 / 0 | 0.008 |
| X site23 sigma600: sync+fov60 vs sync | 400 | 89.2% | 88.0% | +1.2 | -0.2 to +2.8 | 7 / 2 | 0.180 |
| X site23 sigma600: hold vs baseline (sync) | 400 | 88.0% | 88.0% | +0.0 | -1.8 to +1.8 | 6 / 6 | 1.000 |
| X site23 sigma1000: alt_sync vs wide | 400 | 92.2% | 90.0% | +2.2 | +1.0 to +3.8 | 9 / 0 | 0.004 |
| X site23 sigma1000: sync+fov60 vs sync | 400 | 95.5% | 92.2% | +3.2 | +1.5 to +5.2 | 14 / 1 | 0.001 |
| X site23 sigma600: fov20 vs sync (fov40) | 400 | 76.8% | 88.0% | -11.2 | -14.8 to -8.0 | 5 / 50 | 0.000 |
| X site23 sigma600: fov30 vs sync (fov40) | 400 | 85.5% | 88.0% | -2.5 | -4.8 to -0.2 | 6 / 16 | 0.052 |
| X site23 sigma600: fov60 vs fov20 | 400 | 89.2% | 76.8% | +12.5 | +9.0 to +16.2 | 56 / 6 | 0.000 |
| X site23 sigma1000 minus sigma600 (sync, different seeds' difficulty) | 400 | 92.2% | 88.0% | +4.2 | +1.5 to +7.2 | 26 / 9 | 0.006 |
| S: PPO nocov s3 (k=0.5) vs baseline | 200 | 72.5% | 83.0% | -10.5 | -16.0 to -5.0 | 6 / 27 | 0.000 |
| S: PPO nocov s4 (k=0.5) vs baseline | 200 | 76.0% | 83.0% | -7.0 | -12.0 to -2.0 | 7 / 21 | 0.013 |
| S: PPO cov s3 (k=0.5) vs baseline | 200 | 78.5% | 83.0% | -4.5 | -9.5 to +0.0 | 7 / 16 | 0.093 |
| S: PPO cov s4 (k=0.5) vs baseline | 200 | 78.0% | 83.0% | -5.0 | -9.5 to -1.0 | 5 / 15 | 0.041 |
| S: PPO nocov k=1.0 s0 vs baseline | 200 | 82.0% | 83.0% | -1.0 | -5.5 to +3.5 | 9 / 11 | 0.824 |
| S: PPO nocov k=1.0 s1 vs baseline | 200 | 77.0% | 83.0% | -6.0 | -11.0 to -1.0 | 7 / 19 | 0.029 |
| S: PPO nocov k=1.0 s2 vs baseline | 200 | 70.5% | 83.0% | -12.5 | -19.5 to -6.0 | 12 / 37 | 0.000 |
| S: cold-start decision PPO s0 vs baseline (nominal) | 200 | 81.5% | 83.0% | -1.5 | -4.0 to +0.5 | 1 / 4 | 0.375 |
| S: cold-start decision PPO s0 vs hold (nominal) | 200 | 81.5% | 89.0% | -7.5 | -12.0 to -3.0 | 3 / 18 | 0.001 |
| S: cold-start decision PPO s0 vs away-800 rule (nominal) | 200 | 81.5% | 73.5% | +8.0 | +2.5 to +13.5 | 25 / 9 | 0.009 |
| S: cold-start decision PPO s0 vs baseline (realism) | 200 | 73.5% | 73.0% | +0.5 | -1.5 to +2.5 | 3 / 2 | 1.000 |
| S: cold-start decision PPO s0 vs away-800 rule (realism) | 200 | 73.5% | 77.5% | -4.0 | -10.0 to +2.0 | 15 / 23 | 0.256 |
| S: cold-start decision PPO s1 vs baseline (nominal) | 200 | 86.0% | 83.0% | +3.0 | -0.5 to +6.5 | 10 / 4 | 0.180 |
| S: cold-start decision PPO s1 vs hold (nominal) | 200 | 86.0% | 89.0% | -3.0 | -7.5 to +1.5 | 8 / 14 | 0.286 |
| S: cold-start decision PPO s1 vs away-800 rule (nominal) | 200 | 86.0% | 73.5% | +12.5 | +6.5 to +18.5 | 33 / 8 | 0.000 |
| S: cold-start decision PPO s1 vs baseline (realism) | 200 | 78.5% | 73.0% | +5.5 | +1.0 to +10.0 | 15 / 4 | 0.019 |
| S: cold-start decision PPO s1 vs away-800 rule (realism) | 200 | 78.5% | 77.5% | +1.0 | -4.5 to +6.5 | 19 / 17 | 0.868 |
| S: cold-start decision PPO s2 vs baseline (nominal) | 200 | 84.0% | 83.0% | +1.0 | -2.0 to +4.0 | 6 / 4 | 0.754 |
| S: cold-start decision PPO s2 vs hold (nominal) | 200 | 84.0% | 89.0% | -5.0 | -9.5 to -1.0 | 5 / 15 | 0.041 |
| S: cold-start decision PPO s2 vs away-800 rule (nominal) | 200 | 84.0% | 73.5% | +10.5 | +5.0 to +16.5 | 29 / 8 | 0.001 |
| S: cold-start decision PPO s2 vs baseline (realism) | 200 | 78.0% | 73.0% | +5.0 | +2.0 to +8.5 | 11 / 1 | 0.006 |
| S: cold-start decision PPO s2 vs away-800 rule (realism) | 200 | 78.0% | 77.5% | +0.5 | -5.5 to +6.5 | 19 / 18 | 1.000 |
| S: cold-start decision PPO s3 vs baseline (nominal) | 200 | 81.5% | 83.0% | -1.5 | -4.0 to +0.5 | 1 / 4 | 0.375 |
| S: cold-start decision PPO s3 vs hold (nominal) | 200 | 81.5% | 89.0% | -7.5 | -12.0 to -3.0 | 3 / 18 | 0.001 |
| S: cold-start decision PPO s3 vs away-800 rule (nominal) | 200 | 81.5% | 73.5% | +8.0 | +2.5 to +13.5 | 25 / 9 | 0.009 |
| S: cold-start decision PPO s3 vs baseline (realism) | 200 | 73.5% | 73.0% | +0.5 | -1.5 to +3.0 | 3 / 2 | 1.000 |
| S: cold-start decision PPO s3 vs away-800 rule (realism) | 200 | 73.5% | 77.5% | -4.0 | -10.0 to +2.0 | 15 / 23 | 0.256 |
| S: cold-start decision PPO s4 vs baseline (nominal) | 200 | 81.5% | 83.0% | -1.5 | -4.0 to +0.5 | 1 / 4 | 0.375 |
| S: cold-start decision PPO s4 vs hold (nominal) | 200 | 81.5% | 89.0% | -7.5 | -12.0 to -3.0 | 3 / 18 | 0.001 |
| S: cold-start decision PPO s4 vs away-800 rule (nominal) | 200 | 81.5% | 73.5% | +8.0 | +2.5 to +13.5 | 25 / 9 | 0.009 |
| S: cold-start decision PPO s4 vs baseline (realism) | 200 | 73.5% | 73.0% | +0.5 | -1.5 to +2.5 | 3 / 2 | 1.000 |
| S: cold-start decision PPO s4 vs away-800 rule (realism) | 200 | 73.5% | 77.5% | -4.0 | -10.0 to +2.0 | 15 / 23 | 0.256 |
| S: away rule 800 m AT EVERY DECISION vs baseline (nominal) | 200 | 73.5% | 83.0% | -9.5 | -15.5 to -4.0 | 7 / 26 | 0.001 |
| S: away rule 800 m AT EVERY DECISION vs hold (nominal) | 200 | 73.5% | 89.0% | -15.5 | -21.0 to -10.0 | 3 / 34 | 0.000 |
| S: away rule 800 m AT EVERY DECISION vs baseline (realism) | 200 | 77.5% | 73.0% | +4.5 | -1.0 to +10.0 | 22 / 13 | 0.175 |
| S: away rule 800 m AT EVERY DECISION vs hold (realism) | 200 | 77.5% | 78.0% | -0.5 | -6.0 to +5.5 | 17 / 18 | 1.000 |
| S: away rule 400 m AT EVERY DECISION vs baseline (nominal) | 200 | 86.5% | 83.0% | +3.5 | -2.0 to +9.0 | 20 / 13 | 0.296 |
| S: away rule 400 m AT EVERY DECISION vs hold (nominal) | 200 | 86.5% | 89.0% | -2.5 | -7.5 to +2.5 | 11 / 16 | 0.442 |
| S: away rule 400 m AT EVERY DECISION vs baseline (realism) | 200 | 86.5% | 73.0% | +13.5 | +8.0 to +19.0 | 31 / 4 | 0.000 |
| S: away rule 400 m AT EVERY DECISION vs hold (realism) | 200 | 86.5% | 78.0% | +8.5 | +3.5 to +14.0 | 24 / 7 | 0.003 |
| S2: away 800 m first decision only vs baseline (nominal) | 200 | 93.5% | 83.0% | +10.5 | +5.5 to +15.5 | 24 / 3 | 0.000 |
| S2: away 800 m first decision only vs hold (nominal) | 200 | 93.5% | 89.0% | +4.5 | +0.5 to +8.5 | 14 / 5 | 0.064 |
| S2: away 800 m first decision only vs baseline (realism) | 200 | 84.5% | 73.0% | +11.5 | +6.0 to +17.5 | 30 / 7 | 0.000 |
| S2: away 800 m first decision only vs hold (realism) | 200 | 84.5% | 78.0% | +6.5 | +1.5 to +11.5 | 19 / 6 | 0.015 |
| S2: away 400 m first decision only vs baseline (nominal) | 200 | 91.0% | 83.0% | +8.0 | +3.5 to +13.0 | 20 / 4 | 0.002 |
| S2: away 400 m first decision only vs hold (nominal) | 200 | 91.0% | 89.0% | +2.0 | -2.5 to +6.5 | 13 / 9 | 0.523 |
| S2: away 400 m first decision only vs baseline (realism) | 200 | 81.5% | 73.0% | +8.5 | +4.0 to +13.5 | 21 / 4 | 0.001 |
| S2: away 400 m first decision only vs hold (realism) | 200 | 81.5% | 78.0% | +3.5 | -0.5 to +8.0 | 13 / 6 | 0.167 |
| S2: cold-start PPO s0 sampled vs baseline (nominal) | 200 | 87.0% | 83.0% | +4.0 | -0.5 to +8.5 | 16 / 8 | 0.152 |
| S2: cold-start PPO s0 sampled vs hold (nominal) | 200 | 87.0% | 89.0% | -2.0 | -7.0 to +3.0 | 12 / 16 | 0.572 |
| S2: cold-start PPO s0 sampled vs first-only away 800 (nominal) | 200 | 87.0% | 93.5% | -6.5 | -11.5 to -2.0 | 6 / 19 | 0.015 |
| S2: cold-start PPO s0 sampled vs baseline (realism) | 200 | 77.0% | 73.0% | +4.0 | -2.0 to +9.5 | 22 / 14 | 0.243 |
| S2: cold-start PPO s0 sampled vs first-only away 800 (realism) | 200 | 77.0% | 84.5% | -7.5 | -13.5 to -1.5 | 11 / 26 | 0.020 |
| S2: cold-start PPO s1 sampled vs baseline (nominal) | 200 | 93.0% | 83.0% | +10.0 | +4.5 to +15.5 | 27 / 7 | 0.001 |
| S2: cold-start PPO s1 sampled vs hold (nominal) | 200 | 93.0% | 89.0% | +4.0 | -0.5 to +9.0 | 16 / 8 | 0.152 |
| S2: cold-start PPO s1 sampled vs first-only away 800 (nominal) | 200 | 93.0% | 93.5% | -0.5 | -4.0 to +3.0 | 6 / 7 | 1.000 |
| S2: cold-start PPO s1 sampled vs baseline (realism) | 200 | 78.0% | 73.0% | +5.0 | -0.5 to +11.0 | 22 / 12 | 0.121 |
| S2: cold-start PPO s1 sampled vs first-only away 800 (realism) | 200 | 78.0% | 84.5% | -6.5 | -11.5 to -2.0 | 6 / 19 | 0.015 |
| S2: cold-start PPO s2 sampled vs baseline (nominal) | 200 | 90.0% | 83.0% | +7.0 | +2.5 to +11.5 | 18 / 4 | 0.004 |
| S2: cold-start PPO s2 sampled vs hold (nominal) | 200 | 90.0% | 89.0% | +1.0 | -3.5 to +5.5 | 11 / 9 | 0.824 |
| S2: cold-start PPO s2 sampled vs first-only away 800 (nominal) | 200 | 90.0% | 93.5% | -3.5 | -7.5 to +0.5 | 5 / 12 | 0.143 |
| S2: cold-start PPO s2 sampled vs baseline (realism) | 200 | 81.0% | 73.0% | +8.0 | +2.5 to +13.5 | 25 / 9 | 0.009 |
| S2: cold-start PPO s2 sampled vs first-only away 800 (realism) | 200 | 81.0% | 84.5% | -3.5 | -8.5 to +1.5 | 10 / 17 | 0.248 |
| S2: cold-start PPO s3 sampled vs baseline (nominal) | 200 | 87.5% | 83.0% | +4.5 | +0.0 to +9.5 | 17 / 8 | 0.108 |
| S2: cold-start PPO s3 sampled vs hold (nominal) | 200 | 87.5% | 89.0% | -1.5 | -6.0 to +3.0 | 10 / 13 | 0.678 |
| S2: cold-start PPO s3 sampled vs first-only away 800 (nominal) | 200 | 87.5% | 93.5% | -6.0 | -10.0 to -2.0 | 3 / 15 | 0.008 |
| S2: cold-start PPO s3 sampled vs baseline (realism) | 200 | 83.0% | 73.0% | +10.0 | +4.5 to +15.5 | 26 / 6 | 0.001 |
| S2: cold-start PPO s3 sampled vs first-only away 800 (realism) | 200 | 83.0% | 84.5% | -1.5 | -7.5 to +4.0 | 16 / 19 | 0.736 |
| S2: cold-start PPO s4 sampled vs baseline (nominal) | 200 | 90.0% | 83.0% | +7.0 | +3.0 to +11.0 | 16 / 2 | 0.001 |
| S2: cold-start PPO s4 sampled vs hold (nominal) | 200 | 90.0% | 89.0% | +1.0 | -3.0 to +5.0 | 9 / 7 | 0.804 |
| S2: cold-start PPO s4 sampled vs first-only away 800 (nominal) | 200 | 90.0% | 93.5% | -3.5 | -7.0 to +0.0 | 3 / 10 | 0.092 |
| S2: cold-start PPO s4 sampled vs baseline (realism) | 200 | 79.5% | 73.0% | +6.5 | +1.0 to +12.0 | 23 / 10 | 0.035 |
| S2: cold-start PPO s4 sampled vs first-only away 800 (realism) | 200 | 79.5% | 84.5% | -5.0 | -11.0 to +1.0 | 14 / 24 | 0.143 |
