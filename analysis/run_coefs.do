cd \\wsl.localhost\Ubuntu\home\sidlh\Documents\id_frictions\data
use panel_data

tsset gvkey date, quarterly
xtrc euler_equation leverage net_worth var_pct
regress euler_equation net_worth var_pct

bys gvkey: egen sd_leverage = sd(leverage)
bys gvkey: egen sd_networth = sd(net_worth)
bys gvkey: egen sd_varpct = sd(var_pct)

sum sd_*