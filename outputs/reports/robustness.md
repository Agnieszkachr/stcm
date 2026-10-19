# Robustness of the central contrast to the encoder configuration

## primary: ABeZet/Koine-Greek-BERT@b81172ecb9baf65734c88ab65e2eb2491f12f3de (last)

| Set | Space | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess |
|---|---|---|---|---|---|---|
| DT | raw | 0.9504 | 0.8389 | 0.9050 | 0.1115 [0.1021, 0.1200] | 0.0454 |
| TT | raw | 0.9574 | 0.9056 | 0.9460 | 0.0518 [0.0456, 0.0575] | 0.0114 |
| LK | raw | 0.9659 | 0.9071 | 0.9414 | 0.0587 [0.0536, 0.0645] | 0.0245 |
| MK | raw | 0.9686 | 0.9108 | 0.9454 | 0.0578 [0.0527, 0.0626] | 0.0232 |
- DT minus TT excess (raw): +0.0597 [+0.0496, +0.0706], MWU p = 3.16e-14; NN excess +0.0340 [+0.0238, +0.0444]
- DT minus LK excess (raw): +0.0528 [+0.0423, +0.0631], MWU p = 7.73e-13; NN excess +0.0209 [+0.0109, +0.0314]
| DT | cen | 0.7082 | 0.0518 | 0.3815 | 0.6563 [0.6085, 0.7018] | 0.3267 |
| TT | cen | 0.5781 | 0.0529 | 0.3990 | 0.5251 [0.4747, 0.5736] | 0.1791 |
| LK | cen | 0.6512 | 0.0653 | 0.3899 | 0.5859 [0.5395, 0.6347] | 0.2613 |
| MK | cen | 0.6791 | 0.0700 | 0.4079 | 0.6091 [0.5702, 0.6503] | 0.2712 |
- DT minus TT excess (cen): +0.1312 [+0.0633, +0.2003], MWU p = 0.000147; NN excess +0.1476 [+0.0704, +0.2265]
- DT minus LK excess (cen): +0.0704 [+0.0037, +0.1411], MWU p = 0.0292; NN excess +0.0654 [-0.0063, +0.1353]
- Residual after Jaccard (r over all pairs = 0.501): matched +0.1800, mismatched -0.0036, matched - mismatched +0.1836; random-null p = 0.000999, length-null p = 0.000999; DT minus TT on the residual +0.0846 [+0.0205, +0.1465]
- Spearman rho of the double-tradition ranking with the primary configuration: 1.000
- Top five: Serving two masters (0.979); Harvest plentiful (0.955); Lament over Jerusalem (0.944); Hidden from wise revealed (0.915); Return of unclean spirit (0.908)
- Bottom five: Scandals, forgiveness, faith (0.461); Not peace but sword (0.438); Judging twelve tribes (0.435); Parable of Great Banquet (0.205); Narrow gate (0.142)

## last4: ABeZet/Koine-Greek-BERT@b81172ecb9baf65734c88ab65e2eb2491f12f3de (last4)

| Set | Space | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess |
|---|---|---|---|---|---|---|
| DT | raw | 0.9800 | 0.9327 | 0.9613 | 0.0473 [0.0432, 0.0513] | 0.0187 |
| TT | raw | 0.9832 | 0.9615 | 0.9790 | 0.0217 [0.0189, 0.0242] | 0.0042 |
| LK | raw | 0.9869 | 0.9617 | 0.9769 | 0.0252 [0.0230, 0.0277] | 0.0100 |
| MK | raw | 0.9878 | 0.9639 | 0.9785 | 0.0239 [0.0216, 0.0261] | 0.0093 |
- DT minus TT excess (raw): +0.0256 [+0.0212, +0.0305], MWU p = 6.21e-14; NN excess +0.0146 [+0.0100, +0.0193]
- DT minus LK excess (raw): +0.0221 [+0.0176, +0.0268], MWU p = 3.23e-12; NN excess +0.0087 [+0.0045, +0.0133]
| DT | cen | 0.7172 | 0.0617 | 0.4071 | 0.6555 [0.6054, 0.7021] | 0.3101 |
| TT | cen | 0.6007 | 0.0711 | 0.4582 | 0.5296 [0.4768, 0.5785] | 0.1425 |
| LK | cen | 0.6831 | 0.0831 | 0.4441 | 0.5999 [0.5530, 0.6496] | 0.2389 |
| MK | cen | 0.6985 | 0.0947 | 0.4601 | 0.6039 [0.5591, 0.6457] | 0.2384 |
- DT minus TT excess (cen): +0.1258 [+0.0575, +0.1956], MWU p = 0.000287; NN excess +0.1677 [+0.0910, +0.2502]
- DT minus LK excess (cen): +0.0555 [-0.0098, +0.1280], MWU p = 0.0526; NN excess +0.0712 [-0.0036, +0.1443]
- Residual after Jaccard (r over all pairs = 0.479): matched +0.1642, mismatched -0.0033, matched - mismatched +0.1674; random-null p = 0.000999, length-null p = 0.000999; DT minus TT on the residual +0.1856 [+0.1109, +0.2570]
- Spearman rho of the double-tradition ranking with the primary configuration: 0.950
- Top five: Serving two masters (0.983); Harvest plentiful (0.963); Lament over Jerusalem (0.942); Return of unclean spirit (0.924); Thief in the night (0.906)
- Bottom five: Not peace but sword (0.496); Treasure in heaven (0.441); Parable of Great Banquet (0.293); Scandals, forgiveness, faith (0.283); Narrow gate (0.197)

## agb: pranaydeeps/Ancient-Greek-BERT@5e3e29ece1d63029baa226f11105b1e8277c4f07 (last)

| Set | Space | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess |
|---|---|---|---|---|---|---|
| DT | raw | 0.9309 | 0.8059 | 0.8776 | 0.1250 [0.1162, 0.1342] | 0.0533 |
| TT | raw | 0.9425 | 0.8778 | 0.9264 | 0.0647 [0.0573, 0.0715] | 0.0161 |
| LK | raw | 0.9530 | 0.8766 | 0.9192 | 0.0764 [0.0711, 0.0823] | 0.0337 |
| MK | raw | 0.9572 | 0.8831 | 0.9251 | 0.0741 [0.0686, 0.0795] | 0.0321 |
- DT minus TT excess (raw): +0.0604 [+0.0501, +0.0725], MWU p = 2.1e-13; NN excess +0.0372 [+0.0264, +0.0485]
- DT minus LK excess (raw): +0.0486 [+0.0382, +0.0597], MWU p = 2.49e-11; NN excess +0.0196 [+0.0092, +0.0308]
| DT | cen | 0.6777 | 0.0638 | 0.3436 | 0.6139 [0.5716, 0.6565] | 0.3341 |
| TT | cen | 0.5775 | 0.0615 | 0.3627 | 0.5159 [0.4693, 0.5628] | 0.2148 |
| LK | cen | 0.6520 | 0.0682 | 0.3538 | 0.5839 [0.5460, 0.6246] | 0.2982 |
| MK | cen | 0.6785 | 0.0795 | 0.3656 | 0.5990 [0.5637, 0.6371] | 0.3129 |
- DT minus TT excess (cen): +0.0980 [+0.0364, +0.1641], MWU p = 0.00564; NN excess +0.1193 [+0.0553, +0.1893]
- DT minus LK excess (cen): +0.0301 [-0.0252, +0.0932], MWU p = 0.359; NN excess +0.0359 [-0.0250, +0.1038]
- Residual after Jaccard (r over all pairs = 0.499): matched +0.1918, mismatched -0.0038, matched - mismatched +0.1957; random-null p = 0.000999, length-null p = 0.000999; DT minus TT on the residual +0.0883 [+0.0289, +0.1471]
- Spearman rho of the double-tradition ranking with the primary configuration: 0.933
- Top five: Serving two masters (0.977); Ask, seek, knock (0.919); Lament over Jerusalem (0.916); Return of unclean spirit (0.915); Thief in the night (0.899)
- Bottom five: Parable of Great Banquet (0.470); Judging twelve tribes (0.445); Not peace but sword (0.380); Scandals, forgiveness, faith (0.358); Narrow gate (0.305)

## greekbert: nlpaueb/bert-base-greek-uncased-v1@ec2b8f88dd215b5246f2f850413d5bff90d7540d (last)

| Set | Space | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess |
|---|---|---|---|---|---|---|
| DT | raw | 0.9267 | 0.8192 | 0.8899 | 0.1075 [0.0980, 0.1162] | 0.0368 |
| TT | raw | 0.9462 | 0.8991 | 0.9422 | 0.0471 [0.0399, 0.0541] | 0.0040 |
| LK | raw | 0.9580 | 0.8987 | 0.9386 | 0.0593 [0.0543, 0.0646] | 0.0193 |
| MK | raw | 0.9559 | 0.9023 | 0.9401 | 0.0537 [0.0479, 0.0598] | 0.0158 |
- DT minus TT excess (raw): +0.0604 [+0.0493, +0.0724], MWU p = 6.58e-13; NN excess +0.0327 [+0.0217, +0.0441]
- DT minus LK excess (raw): +0.0482 [+0.0380, +0.0591], MWU p = 2.89e-11; NN excess +0.0174 [+0.0069, +0.0283]
| DT | cen | 0.6232 | 0.0529 | 0.3793 | 0.5702 [0.5208, 0.6170] | 0.2439 |
| TT | cen | 0.5234 | 0.0805 | 0.4262 | 0.4430 [0.3906, 0.4940] | 0.0972 |
| LK | cen | 0.6248 | 0.0949 | 0.4267 | 0.5300 [0.4850, 0.5762] | 0.1981 |
| MK | cen | 0.5966 | 0.0873 | 0.4106 | 0.5093 [0.4590, 0.5587] | 0.1859 |
- DT minus TT excess (cen): +0.1273 [+0.0605, +0.2018], MWU p = 0.000836; NN excess +0.1467 [+0.0721, +0.2239]
- DT minus LK excess (cen): +0.0403 [-0.0235, +0.1089], MWU p = 0.217; NN excess +0.0458 [-0.0266, +0.1196]
- Residual after Jaccard (r over all pairs = 0.411): matched +0.1597, mismatched -0.0032, matched - mismatched +0.1629; random-null p = 0.000999, length-null p = 0.000999; DT minus TT on the residual +0.1426 [+0.0712, +0.2087]
- Spearman rho of the double-tradition ranking with the primary configuration: 0.836
- Top five: Serving two masters (0.970); Anxieties about life (0.894); Harvest plentiful (0.877); Ask, seek, knock (0.856); Hidden from wise revealed (0.854)
- Bottom five: Law, prophets and divorce (0.412); Blind guides (0.398); House on the rock (0.236); Narrow gate (0.164); Scandals, forgiveness, faith (0.122)
