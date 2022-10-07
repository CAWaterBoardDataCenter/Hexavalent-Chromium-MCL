import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

data_folder = 'Data/'
results_folder = 'Results/'
specific_results_folder = '22-09-28 for SRIA/'
if not os.path.exists(results_folder+specific_results_folder):
    os.makedirs(results_folder+specific_results_folder)
file = 'Raw_Data_edited_21-07-27.csv' # contaminant data
TNC_file = 'NC_Cr6_Data.csv'
EAR_file = 'Master_WaterVol_inGallons_21-02-19.csv'
DLR = 1
Calc_Treatment = True  # Calculate Treatment Costs
Tabulate = True # to make bins and final table
Use_EAR_data = False # to use EAR data, when available to estimate system flow
suffix = '_22-09-28_highest-yr_Jun2022$'
Finding_Type = 'highest year' # 'average all' or 'average year' or 'highest year'
sampling_cost = 78.63
PHG = 0.02
#######################################################################
## Unit Costs ##
unit_energy_cost = 0.193 # $ per kWh
ferrous_sulfate_unit_cost = 9.09 # $ per gallon
salt_unit_cost = 192.6775 # $ per ton
sludge_disposal_unit_cost = 2583 # $ per ton
brine_disposal_unit_cost = 297.081 # $ per kgal
SBA_resin_unit_cost = 243.657 # $ per cf
WBA_resin_unit_cost = 600 # $ per cf 
HCl_unit_cost = 2791.84 # $ per ton
NaOH_unit_cost = 1269.02 # $ per ton

## Get Data and Initialize ##
OG_Data = pd.read_csv(data_folder+file,header=0,encoding='cp437')
NC_Data = pd.read_csv(data_folder+TNC_file,header=0)
EAR_Data = pd.read_csv(data_folder+EAR_file,header=0)
Arsenic_Data = pd.read_csv(data_folder+'Arsenic_Facilities.csv',header=0)
Sulfate_Data = pd.read_csv(data_folder+'Sulfate_22-01-24.csv',skiprows=2,header=0,encoding='cp437')
Nitrate_Data = pd.read_csv(data_folder+'Nitrate_22-07-08.csv',skiprows=2,header=0,encoding='cp437')

NC_Data.rename(columns={'Water System #':'Water System'},inplace=True)
OG_Data = pd.concat([OG_Data,NC_Data])

Master = pd.read_csv(data_folder+'Master_21-07-06_whol-pop.csv',header=0) # Master Spreadsheet with all source and system data
Master['Overall Finding'] = np.nan # set up average or annual finding column
Potential_MCLs = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','20','25','30','35','40','45']
Data_Changes = {'0110003':0.4,'0110010':0.15,'0710002':0.15,'3910001':0.19,'3610029':0.5,'3610018':0.5,'1910048':0.054,'1910067':0.087,
                '3610041':0.5,'3610055':0.5,'3610037':0.5,'3810011':0.004523,'3910011':0.043,'3910012':0.17,'5010005':0.48,'5010010':0.2,
                '3810001':0.0002}
Well_Changes = ['3410020','3410017','3410029','5710006','5710001','5710009']
Source_Columns = []
new_columns = []
final_table = []
for MCL in Potential_MCLs:
    MCL_int = int(MCL)
    Source_Columns.append('No. of Sources SBA_'+MCL)
    Master['MCL-'+MCL+'_SBA_Cap_HS'] = np.nan
    Master['MCL-'+MCL+'_SBA_OM_HS'] = np.nan
    ## SBA
    new_columns.append('MCL-'+MCL+'_monitoring')
    new_columns.append('MCL-'+MCL+'_Compliance_Plan')
    new_columns.append('MCL-'+MCL+'_SBA_Cap')
    new_columns.append('MCL-'+MCL+'_SBA_OM')
    new_columns.append('MCL-'+MCL+'_SBA_OM-disp')
    new_columns.append('MCL-'+MCL+'_SBA_OM-resin')
    new_columns.append('MCL-'+MCL+'_SBA_OM-chem')
    new_columns.append('MCL-'+MCL+'_SBA_OM-labor')
    new_columns.append('MCL-'+MCL+'_SBA_OM-energy')
    new_columns.append('MCL-'+MCL+'_SBA_designf_gpm')
    new_columns.append('MCL-'+MCL+'_SBA_newflow_gpm')
    # new_columns.append('MCL-'+MCL+'_SBA_OM-BVs')
    ## RCF
    new_columns.append('MCL-'+MCL+'_RCF_Cap')
    new_columns.append('MCL-'+MCL+'_RCF_OM')
    new_columns.append('MCL-'+MCL+'_RCF_OM-disp')
    new_columns.append('MCL-'+MCL+'_RCF_OM-chem')
    new_columns.append('MCL-'+MCL+'_RCF_OM-labor')
    new_columns.append('MCL-'+MCL+'_RCF_OM-energy')
    new_columns.append('MCL-'+MCL+'_RCF_designf_gpm')
    new_columns.append('MCL-'+MCL+'_RCF_newflow_gpm')
    ## WBA
    new_columns.append('MCL-'+MCL+'_WBA_Cap')
    new_columns.append('MCL-'+MCL+'_WBA_OM')
    new_columns.append('MCL-'+MCL+'_WBA_OM-disp')
    new_columns.append('MCL-'+MCL+'_WBA_OM-resin')
    new_columns.append('MCL-'+MCL+'_WBA_OM-chem')
    new_columns.append('MCL-'+MCL+'_WBA_OM-labor')
    new_columns.append('MCL-'+MCL+'_WBA_OM-energy')
    new_columns.append('MCL-'+MCL+'_WBA_designf_gpm')
    new_columns.append('MCL-'+MCL+'_WBA_newflow_gpm')
    ## Annual Costs
    new_columns.append('MCL-'+MCL+'_Ann_SBA')
    new_columns.append('MCL-'+MCL+'_Ann_WBA')
    new_columns.append('MCL-'+MCL+'_Ann_RCF')
    ## Chosen
    new_columns.append('MCL-'+MCL+'_Treatment_chosen')
    new_columns.append('MCL-'+MCL+'_Annualized_cost')
    new_columns.append('MCL-'+MCL+'_Cap_Cost_chosen-tot')
    new_columns.append('MCL-'+MCL+'_Cap_Cost_chosen-ann')
    new_columns.append('MCL-'+MCL+'_OM_Cost_chosen')
    new_columns.append('MCL-'+MCL+'_Chosen_disposal')
    new_columns.append('MCL-'+MCL+'_Chosen_resin')
    new_columns.append('MCL-'+MCL+'_Chosen_energy')
    new_columns.append('MCL-'+MCL+'_Chosen_chem')
    new_columns.append('MCL-'+MCL+'_Chosen_labor')
    new_columns.append('MCL-'+MCL+'_Chosen_designf_MG')
    new_columns.append('MCL-'+MCL+'_Chosen_flow_MG')
    new_columns.append('MCL-'+MCL+'_Risk')
    new_columns.append('MCL-'+MCL+'_Avoided_Cancer_Cases')

new_columns.append('Nitrate Avg (mg/L)')
new_columns.append('Sulfate Avg (mg/L)')

all_the_columns = list(OG_Data.columns)
for x in new_columns:
    all_the_columns.append(x)

def Interpolate(flow_options,cost_dict,design_f):
    try:
        interp_cost = cost_dict[design_f]
        return interp_cost
    except:
        top = 0
        bottom = 0
        for flow_dict in range(0,len(flow_options)):
            if design_f < flow_options[flow_dict]:
                top = flow_options[flow_dict]
                bottom = flow_options[flow_dict-1]
                break
        bottom_diff = design_f-bottom
        flow_diff = top-bottom
        cost_diff = cost_dict[str(top)] - cost_dict[str(bottom)]
        interp_cost = cost_dict[str(bottom)] + (bottom_diff/flow_diff)*(cost_diff)
        return interp_cost


def Hexavalent_Chromium_Treatment_Costs_SBA(MCL,gallons,finding,service_connections,sulfate_avg,nitrate_avg):
    cap_costs_list_smalls = {'10':222267,'20':242612,'40':326928,'60':333153,'80':339167,'100':355203}
    cap_costs_list_4450 = {'100':1936708.77,'250':3241901.93,'500':4411782.44,'1000':5642871.09,'2000':8921148.48,'5000':13193545.36,'7500':18826592.27,'10000':22704231.05} # in 2022$
    flow_options_EPA = [10,20,40,70,100,150,200,350,694,695,800,1000,1500,2000,3000,5000,10000,15000,20000,25000]
    flow_options_smalls = [10,20,40,60,80,100]
    flow_options_4450 = [100,250,500,1000,2000,5000,7500,10000]
    peaking_factor = 1.5
    cap_costs = 0
    OM_costs = 0
    MCL_int = int(MCL)
    design_f = gallons/365.25/24/60*peaking_factor # to design flow gpm
    if design_f < 10:
       design_f = 10

    avg_f_gpm = design_f/peaking_factor # in gpm
    avg_f_gal = gallons

    ## By-pass ##
    avg_f_bypass_gal = gallons * (finding - MCL_int*0.8)/(finding - 0.8) # in gallons # flow to be treated after bypass is installed (not bypass flow)
    # avg_f_bypass_L = avg_f_bypass_gal*3.78541 # in liters 
    avg_f_bypass_gpm = avg_f_bypass_gal/365.25/24/60

    BV = 6541 - 8307.5*np.log10((sulfate_avg/48)+(nitrate_avg/62))
    if math.isnan(BV):
        BV = 6400 # conservative assumption (WRF 4450)
    if BV < 500:
        BV = 500 # min BV, even with difficult water quality
    BV_Factor = BV/6400 # more or fewer BVs to breakthrough based on WQ

    ## WRF 4450 O&M Estimates ##
    # 60% utilization rate already accounted for; below data is for 100% utilization
    salt_tons = {'0':0,'10':0.5475,'100':5.475,'250':13.688,'500':27.375,'1000':54.933,'2000':123.553,'5000':352.955,'7500':561.37,'10000':760.113}
    ferrous_gal = {'0':0,'10':41.6,'100':728,'250':1872,'500':3744,'1000':7488,'2000':16848,'5000':48100,'7500':76492,'10000':103584}
    energy_kWh = {'0':0,'10':378.67,'100':3802,'250':9507,'500':19012,'1000':38023,'2000':76048,'5000':190120,'7500':285180,'10000':380240}
    Cl_Wst_Brine_gal = {'0':0,'10':2444,'100':24596,'250':61516,'500':123032,'1000':246064,'2000':553644,'5000':1581788,'7500':2516488,'10000':3406988}
    Sludge_tons = {'0':0,'10':0.234,'100':2.496,'250':6.266,'500':12.506,'1000':25.012,'2000':56.29,'5000':160.81,'7500':255.84,'10000':346.372}
    LG_resin_vol_ft3 = {'0':0,'10':10,'100':64,'250':254,'500':452,'1000':707,'2000':1357,'5000':2375,'7500':3732,'10000':4411}
    SM_resin_vol_ft3 = {'0':0,'10':9,'20':9,'40':32,'60':32,'80':32,'100':40}
    flow_options_LG = [0,10,100,250,500,1000,2000,5000,7500,10000]
    flow_options_SM = [0,10,20,40,60,80,100]

    if design_f >= 100:
        ## Large System Capital Costs using WRF 4450 ## 
        cap_costs = Interpolate(flow_options_4450,cap_costs_list_4450,design_f) # already in 2022$
        initial_resin_vol = Interpolate(flow_options_LG,LG_resin_vol_ft3,design_f)
        initial_resin_cost = initial_resin_vol*SBA_resin_unit_cost*1.125 # includes tax on resin and contractor installation
        tot_cap_costs = cap_costs + initial_resin_cost
        
        ## Large System O&M Costs using WRF 4450 ##
        salt = Interpolate(flow_options_LG,salt_tons,avg_f_bypass_gpm)
        salt_cost = salt*salt_unit_cost 
        ferrous = Interpolate(flow_options_LG,ferrous_gal,avg_f_bypass_gpm)
        ferrous_cost = ferrous*ferrous_sulfate_unit_cost
        brine = Interpolate(flow_options_LG,Cl_Wst_Brine_gal,avg_f_bypass_gpm)
        brine_cost = brine/1000*brine_disposal_unit_cost # convert to kgal before applying cost 
        sludge = Interpolate(flow_options_LG,Sludge_tons,avg_f_bypass_gpm)
        sludge_cost = sludge*sludge_disposal_unit_cost
        resin_vol = Interpolate(flow_options_LG,LG_resin_vol_ft3,avg_f_bypass_gpm)
        resin_cost = resin_vol*.1*SBA_resin_unit_cost # assumes 10% of resin vol is replaced annually
        
        ## Divide by BV Factor to account for regeneration frequency (based on WQ) ##
        salt_cost = salt_cost/BV_Factor
        ferrous_cost = ferrous_cost/BV_Factor
        brine_cost = brine_cost/BV_Factor
        sludge_cost = sludge_cost/BV_Factor

        brine_cost = brine_cost/2 # brine can be reused twice with no adverse affects on BV
        disposal_cost = brine_cost + sludge_cost
        labor_cost = 105798
        
    else:
        ## Small System Capital Costs using EPA 2021 ##
        cap_costs = Interpolate(flow_options_smalls,cap_costs_list_smalls,design_f)
        cap_costs = cap_costs*13110.5/12464.55 # to June2022$ from Sept2021$ using ENR construction index
        initial_resin_vol = Interpolate(flow_options_SM,SM_resin_vol_ft3,design_f)
        initial_resin_cost = initial_resin_vol*SBA_resin_unit_cost*1.125 # includes tax on resin and contractor installation
        tot_cap_costs = cap_costs + initial_resin_cost

        ## Small System O&M Costs using modified WRF 4450 ##
        resin_vol = Interpolate(flow_options_SM,SM_resin_vol_ft3,avg_f_bypass_gpm)
        time_to_spent = BV*3 # in minutes
        no_regens_per_year = 365*24*60/time_to_spent
        # print('Times resin replaced per year:',no_regens_per_year)
        new_resin_cf = resin_vol*no_regens_per_year
        # print('New Resin (cf):',new_resin_cf)
        resin_cost = new_resin_cf*SBA_resin_unit_cost
        # print('Resin Cost:',resin_cost)
        spent_resin = new_resin_cf*43/2000 # 43 pounds per cf # need in per ton for unit disposal cost
        disposal_cost = spent_resin*sludge_disposal_unit_cost
        # print('Disposal Cost:',disposal_cost)
        labor_cost = 105798/2
        salt_cost = 0
        ferrous_cost = 0

    energy = Interpolate(flow_options_LG,energy_kWh,avg_f_bypass_gpm)
    energy_cost = energy*unit_energy_cost
    maintenance_cost = cap_costs*.03 # 3% of cap costs (w/o resin)

    OM_costs = labor_cost + disposal_cost + energy_cost + salt_cost + ferrous_cost + resin_cost + maintenance_cost

    return tot_cap_costs,OM_costs,disposal_cost,resin_cost,BV,energy_cost,ferrous_cost,labor_cost,design_f,avg_f_bypass_gpm


def Hexavalent_Chromium_Treatment_Costs_RCF(MCL,gallons,finding,service_connections):
    ## Costs (already in 2022 dollars) ##
    equip_costs = {'100':464263.61, '250':737663.28, '500':1232877.8, '1000':1563020.81, '2000':2016279.64, '5000':3930765.19, '7500':5733827.48, '10000':7201588.27} 
    build_constr_costs = {'100':552393.65, '250':837373.24, '500':1352807.15, '1000':1690289.83, '2000':2161785, '5000':4156159.8, '7500':6144319.08, '10000':7629237.52}
    initial_services_costs = {'100':367891.84, '250':571246.31, '500':943947.47, '1000':1193170.93, '2000':1560957.13, '5000':3055018.13, '7500':4492666.37, '10000':5631583.89} # professional services and initial filtration media load
    energy_kWh = {'100':2281, '250':5704, '500':11407, '1000':22814, '2000':45629, '5000':114072, '7500':171108, '10000':228144} # in kWh per year
    ferrous_gal = {'100':2040, '250':5099, '500':10198, '1000':20397, '2000':40794, '5000':101984, '7500':152976, '10000':203968} # in gallons per year
    disposal_tons = {'100':2.5116, '250':6.2712, '500':12.5268, '1000':25.0536, '2000':50.1228, '5000':125.2836, '7500':187.9332, '10000':250.5828}
    flow_options_LG = [100,250,500,1000,2000,5000,7500,10000]
    flow_options_SM = [1,5,10,20,50,100]
    SM_cap_costs = {'5':225000,'10':237500,'20':250000,'50':606250,'100':731250} #includes contingency
    SM_OM_costs = {'5':25000,'10':30000,'20':40000,'50':95000,'100':155000}
    SM_energy_kWh = {'5':1095,'10':2190,'20':4380,'50':54750,'100':109500}

    ## to account for utilization rate of 60% in O&M costs ##
    energy_kWh.update((x,y/0.6) for x, y in energy_kWh.items())
    ferrous_gal.update((x,y/0.6) for x, y in ferrous_gal.items())
    disposal_tons.update((x,y/0.6) for x, y in disposal_tons.items())

    ## Flow Calculations ##
    peaking_factor = 1.5
    cap_costs = 0   
    OM_costs = 0
    MCL_int = int(MCL)
    design_f = gallons/365.25/24/60*peaking_factor # to design flow gpm
    avg_f_gpm = design_f/peaking_factor
    ## By-pass ##
    avg_f_bypass = avg_f_gpm * (finding - MCL_int*0.8)/(finding - 0.8) # in gpm # flow to be treated after bypass is installed (not bypass flow)

    if design_f < 5:
        design_f = 5

    if avg_f_bypass <5:
        avg_f_bypass = 5

    if design_f < 100:
        cap_costs = Interpolate(flow_options_SM,SM_cap_costs,design_f)
        OM_costs = Interpolate(flow_options_SM,SM_OM_costs,avg_f_bypass)
        energy = Interpolate(flow_options_SM,SM_energy_kWh,avg_f_bypass)
        energy_cost = energy*unit_energy_cost
        disposal_cost = 0
        chemicals_cost = 0
        labor_cost = 52899
        OM_costs = OM_costs + energy_cost

    else:
        ## Capital Costs ##
        cap_costs_equip = Interpolate(flow_options_LG,equip_costs,design_f)
        cap_costs_building_constr = Interpolate(flow_options_LG,build_constr_costs,design_f)
        init_services = Interpolate(flow_options_LG,initial_services_costs,design_f)
        cap_costs = cap_costs_equip + cap_costs_building_constr + init_services
        
        ## O&M Costs ##
        energy = Interpolate(flow_options_LG,energy_kWh,avg_f_bypass)
        energy_cost = energy*unit_energy_cost
        ferrous = Interpolate(flow_options_LG,ferrous_gal,avg_f_bypass)
        chemicals_cost = ferrous*ferrous_sulfate_unit_cost
        disposal = Interpolate(flow_options_LG,disposal_tons,avg_f_bypass)
        disposal_cost = disposal*sludge_disposal_unit_cost
        labor_cost = 105798
        maintenance_cost = 0.03*(cap_costs_equip+cap_costs_building_constr) # 3% of equipment, building, and construction costs
        OM_costs = energy_cost + chemicals_cost + disposal_cost + labor_cost + maintenance_cost

    return cap_costs,OM_costs,disposal_cost,chemicals_cost,energy_cost,labor_cost,design_f,avg_f_bypass


def Hexavalent_Chromium_Treatment_Costs_WBA(MCL,gallons,finding,service_connections):
    cap_costs_list_4450 = {'100':1073296,'250':1554311,'500':2082033,'1000':3759558,'2000':6520844,'5000':9662353,'7500':13324763,'10000':17276495} # already converted to 2022 $
    flow_options_LG = [100,250,500,1000,2000,5000,7500,10000]
    ## O&M Data ##
    HCl_tons = {'0':0,'100':21.67,'250':53.33,'500':108.33,'1000':216.67,'2000':431.67,'5000':1080,'7500':1618.33,'10000':2158.33}
    NaOH_tons = {'0':0,'100':23.33,'250':58.33,'500':116.67,'1000':231.67,'2000':463.33,'5000':1160,'7500':1740,'10000':2320}
    Energy_kWh = {'0':0,'100':3801.67,'250':9506.67,'500':19011.67,'1000':38023.33,'2000':76048.33,'5000':190120,'7500':285180,'10000':380240}
    Resin_replaced_ft3 = {'0':0,'100':35,'250':86.67,'500':171.67,'1000':345,'2000':690,'5000':1723.33,'7500':2586.67,'10000':3448.33}
    resin_utl_rate = finding/24 #ft3/MG

    peaking_factor = 1.5
    cap_costs = 0
    OM_costs = 0
    MCL_int = int(MCL)
    design_f = gallons/365.25/24/60*peaking_factor # to design flow gpm
    
    ## By-pass ##
    avg_f_bypass_gal = gallons * (finding - MCL_int*0.8)/(finding - 0.8) # in gallons # flow to be treated after bypass is installed (not bypass flow)
    avg_f_bypass_MG = avg_f_bypass_gal/1000000
    # avg_f_bypass_L = avg_f_bypass_gal*3.78541 # in liters 
    avg_f_bypass_gpm = avg_f_bypass_gal/365.25/24/60

    if design_f >= 100:
        ## Capital Costs ##
        cap_costs = Interpolate(flow_options_LG,cap_costs_list_4450,design_f)

        ## O&M Costs ##
        HCl = Interpolate(flow_options_LG,HCl_tons,avg_f_bypass_gpm)
        HCl_cost = HCl*HCl_unit_cost
        NaOH = Interpolate(flow_options_LG,NaOH_tons,avg_f_bypass_gpm)
        NaOH_cost = NaOH*NaOH_unit_cost
        energy = Interpolate(flow_options_LG,Energy_kWh,avg_f_bypass_gpm)
        energy_cost = energy*unit_energy_cost
        # resin_OG = Interpolate(flow_options_LG,Resin_replaced_ft3,avg_f_bypass_gpm)
        resin_calc = resin_utl_rate*avg_f_bypass_MG
        resin_init_cost = resin_calc*WBA_resin_unit_cost
        resin_cost = resin_init_cost*1.175 # taxes and install cost
        labor_cost = 105798
        maintenance_cost = cap_costs*.03 # 3% of cap costs
        disposal_cost = resin_calc*sludge_disposal_unit_cost
    
    else:
        cap_costs = 0
        labor_cost = 0
        disposal_cost = 0
        energy_cost = 0
        resin_cost = 0
        HCl_cost = 0
        NaOH_cost = 0
        maintenance_cost = 0

    OM_costs = labor_cost + disposal_cost + energy_cost + resin_cost + HCl_cost + NaOH_cost + maintenance_cost

    return cap_costs,OM_costs,disposal_cost,resin_cost,energy_cost,HCl_cost,NaOH_cost,labor_cost,design_f,avg_f_bypass_gpm


def Most_Recent_Year_Average(data):
    data['Sample Date'] = pd.to_datetime(data['Sample Date'])
    data.sort_values(by='Sample Date',ascending=True,inplace=True)
    Findings = data['Finding']
    data.index = data['Sample Date']
    month_no = data.index[-1].month

    if month_no == 1:
        month = 'FEB'
    if month_no == 2:
        month = 'MAR'
    if month_no == 3:
        month = 'APR'
    if month_no == 4:
        month = 'MAY'
    if month_no == 5:
        month = 'JUN'
    if month_no == 6:
        month = 'JUL'
    if month_no == 7:
        month = 'AUG'
    if month_no == 8:
        month = 'SEP'
    if month_no == 9:
        month = 'OCT'
    if month_no == 10:
        month = 'NOV'
    if month_no == 11:
        month = 'DEC'
    if month_no == 12:
        month = 'JAN'

    A_Finding = data.resample('AS-'+month).mean()['Finding'][-1]
    return A_Finding

def Highest_Average_Year(data): # Compliance: converts raw data to average quartely data, then evaluates rolling 1yr average for highest instance
    data['Sample Date'] = pd.to_datetime(data['Sample Date'])
    data.sort_values(by='Sample Date',ascending=True,inplace=True)
    data.index = data['Sample Date']
    data = data['Finding'].resample('Q').mean()
    data = data.rolling(window=4,min_periods=1).mean(skipna=True)

    after_max = data.loc[data.idxmax()::]
    try:
        after_max = after_max.iloc[3::]
        after_max.dropna(axis=0,inplace=True)
        if (after_max.max() == 1) and (data.max() > 10):# < data.max()/5:
            print('----OUTLIER----')
            print(data)
            print(after_max)
            print(data.max())
    except:
        pass    

    A_Finding = data.max()
    return A_Finding

def Drop_Unwanted_Sources(data):
    source_facilities = ['WL','SP','IN','CH','IG'] # no 'RS'
    name = data.columns.get_loc('Facility Name')
    facility = data.columns.get_loc('Facility Type')
    fac_status_index = data.columns.get_loc('Facility Status')
    system_index = data.columns.get_loc('Water System')
    description_index = data.columns.get_loc('Description')
    removals = ['STANDBY','STAND-BY','STNB','STNDB','INAC','inac','DEAD','dead','DESTROYED','SEASONAL','emergency','EMERGENCY']
    rows_to_drop = []
    for x in range(0,len(data)):
        for removal in removals:
            if removal in str(data.iloc[x,name]):
                if 'DEADWOOD' in str(data.iloc[x,name]):
                    pass
                else:
                    rows_to_drop.append(x)
        if data.iloc[x,facility] not in source_facilities:
            if x not in rows_to_drop:
                rows_to_drop.append(x)
        if data.iloc[x,fac_status_index] == 'I':
            if x not in rows_to_drop:
                rows_to_drop.append(x)
        if data.iloc[x,data.columns.get_loc('Note 3')] == 'STBY':
            if x not in rows_to_drop:
                rows_to_drop.append(x)
        if 'STANDBY' in str(data.iloc[x,description_index]): # drop standby wells
            if x not in rows_to_drop:
                rows_to_drop.append(x)
        elif 'STBY' in str(data.iloc[x,description_index]): # drop standby wells
            if x not in rows_to_drop:
                rows_to_drop.append(x)
    dropped_data = pd.DataFrame(columns=data.columns)
    new_data = pd.DataFrame(columns=data.columns)
    for x in range(0,len(data)):
        if x in rows_to_drop:
            dropped_data = dropped_data.append(data.iloc[x,:])
        else:
            new_data = new_data.append(data.iloc[x,:])
    dropped_data.to_csv(data_folder+'Dropped_data_%s.csv'%suffix)
    return new_data

def Calculate_All_Treatment(Potential_MCLs,data,Master,suffix,all_the_columns,Finding_Type,DLR,List_Large=False):
    ## Get a set of sources in order ##
    sources = data['PS Code'].sort_values(ascending=True) # get source IDs in order
    set_OG_Data = pd.DataFrame(columns=data.columns)
    source_set_OG = []
    for i,source in enumerate(sources):
        if source not in source_set_OG:
            source_set_OG.append(source)
            set_OG_Data = set_OG_Data.append(data.iloc[i,:])

    ## Set up New Set for Final Data Collection #
    NEW_SET = pd.DataFrame(columns=all_the_columns)
    NEW_SET['Overall Finding'] = np.nan
    NEW_SET['Number of Sources'] = np.nan
    NEW_SET['System Flow (G)'] = np.nan
    NEW_SET['Source Flow (G)'] = np.nan
    ann_finding_index = NEW_SET.columns.get_loc('Overall Finding')
    num_sources_index = NEW_SET.columns.get_loc('Number of Sources')
    system_flow_index = NEW_SET.columns.get_loc('System Flow (G)')
    source_flow_index = NEW_SET.columns.get_loc('Source Flow (G)')
    pop_index = Master.columns.get_loc('Population')
    sc_index = Master.columns.get_loc('Service Connections')
    source_type_index = NEW_SET.columns.get_loc('Water Type Code')

    ## Calculate Treatment Costs ##
    minus = 0
    for i,source in enumerate(source_set_OG):
        collect = pd.DataFrame()
        m_index = np.nan
        new_i = i - minus
        source_sys_no = source[0:7]
        if 'CA4400774' in source_sys_no:
            print('----AT CA4400774')
        for m in range(0,len(Master)):
            m_ps_code = str(Master.iloc[m,Master.columns.get_loc('Water System No.')]).zfill(7) #+'-'+str(Master.iloc[m,2]).zfill(3)
            if m_ps_code == source_sys_no: # find index for matching source entry in Master
                m_index = m
                continue
        print(source_sys_no)
        if math.isnan(m_index):
            minus += 1
            continue
        ## Get all finding data for each source ##
        for k,source_x in enumerate(data['PS Code']):
            if source == source_x:
                collect = collect.append(data.iloc[k,:])
        collect.loc[(collect.Finding < DLR),'Finding'] = DLR

        if len(collect) > 0: # data is found
            NEW_SET = NEW_SET.append(collect.iloc[0,:])
            if len(collect) > 1:
                if Finding_Type == 'average year':
                    A_Finding = Most_Recent_Year_Average(collect) # averages only most recent year of data
                elif Finding_Type == 'average all':
                    A_Finding = collect['Finding'].mean() # averages all finding data
                elif Finding_Type == 'highest year':
                    A_Finding = Highest_Average_Year(collect) # gets highest average year
                else:
                    print('--FINDING TYPE ERROR--')
                if A_Finding > 50:
                    A_Finding = 50
                NEW_SET.iloc[new_i,ann_finding_index] = A_Finding
            else: # only one finding value
                A_Finding = float(collect['Finding'])
                if A_Finding > 50:
                    A_Finding = 50
                NEW_SET.iloc[new_i,ann_finding_index] = A_Finding
            ## Find sulfate and nitrate concentrations ##
            if A_Finding > 1: # we only need this info if we're going to calculate treatment costs
                source2 = 'CA'+source+source[-4:]
                source2 = source2.replace('-','_')
                S_data = Sulfate_Data[Sulfate_Data['PS CODE']==source2]
                Sulfate_avg = S_data['Result'].mean()
                N_data = Nitrate_Data[Nitrate_Data['PS CODE']==source2]
                Nitrate_avg = N_data['Result'].mean()
                NEW_SET.iloc[new_i,NEW_SET.columns.get_loc('Nitrate Avg (mg/L)')] = Nitrate_avg
                NEW_SET.iloc[new_i,NEW_SET.columns.get_loc('Sulfate Avg (mg/L)')] = Sulfate_avg

            ## Get Amount of Water and Calc Treatment Costs ##    
            for MCL in Potential_MCLs:
                monit_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_monitoring')
                compliance_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Compliance_Plan')
                
                if round(NEW_SET.iloc[new_i,ann_finding_index],4) > int(MCL): # if finding value is greater than potential MCL
                    # print(NEW_SET.iloc[new_i,ann_finding_index],int(MCL))
                    ## Compliance Plan Costs ##
                    try:
                        system_num = NEW_SET.iloc[new_i,NEW_SET.columns.get_loc('Water System')]
                        check_sys_set = NEW_SET.loc[NEW_SET['Water System']==system_num]
                    except:
                        system_num = NEW_SET.iloc[new_i,NEW_SET.columns.get_loc('Water System #')]
                        check_sys_set = NEW_SET.loc[NEW_SET['Water System #']==system_num]
                    total_compliance = check_sys_set['MCL-'+MCL+'_Compliance_Plan'].sum()
                    if total_compliance > 0:
                        compliance_cost = 0
                    else:
                        compliance_cost = 7619.23
                    NEW_SET.iloc[new_i,compliance_index] = compliance_cost

                    ## Monitoring Costs ##
                    monitoring_cost = (12+4)*sampling_cost
                    NEW_SET.iloc[new_i,monit_index] = monitoring_cost

                    ## Check for Data Changes ##
                    Water_diff = 1
                    for change_sys in Data_Changes:
                        if source_sys_no == change_sys:
                            Water_diff = Data_Changes[change_sys]
                            # print('--changed',Water_diff)
                    well_cap = False
                    for change_sys in Well_Changes:
                        if source_sys_no == change_sys:
                            well_cap = True
                            # print('--capped')
                    ## Special Cases ##
                    if source == '1910097-011':
                        # print('---WORKED for CA1910097')
                        Water_diff = 0.1
                    if source == '3910011-019':
                        # print('---WORKED FOR CA3910011')
                        Water_diff = 1
                    if source == '3910012-102':
                        # print('---worked again')
                        Water_diff = 1
                    if source == '3910012-103':
                        # print('---worked again again')
                        Water_diff = 1
                    if source == '3610036-014':
                        # print('---should remove source')
                        continue

                    EAR_Water = 0
                    if Use_EAR_data:
                        print('------------------ USING EAR DATA -------------------')
                        if Master.iloc[m_index,pop_index] < 2:
                            for i in range(0,len(EAR_Data)):
                                print(source_sys_no)
                                if EAR_Data.iloc[i,EAR_Data.columns.get_loc('Water System No.')] == source_sys_no:
                                    print('--yes')
                                    if EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2019')] > 0:
                                        EAR_Water = EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2019')]
                                    if EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2018')] > 0:
                                        EAR_Water = EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2018')]
                                    if EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2017')] > 0:
                                        EAR_Water = EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2017')]
                                    if EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2016')] > 0:
                                        EAR_Water = EAR_Data.iloc[i,EAR_Data.columns.get_loc('Annual GW 2016')]

                    if EAR_Water == 0:
                        gpcpd = np.nan
                        if NEW_SET.iloc[0,NEW_SET.columns.get_loc('FED Type')] == 'NC':
                            gpcpd = 120
                        elif NEW_SET.iloc[0,NEW_SET.columns.get_loc('FED Type')] == 'NTNC':
                            gpcpd = 120
                        elif NEW_SET.iloc[0,NEW_SET.columns.get_loc('FED Type')] == 'C   ':
                            gpcpd = 150
                        else: 
                            print('-----ERROR: Water Production')
                            print(NEW_SET.iloc[0,NEW_SET.columns.get_loc('FED Type')])
                        # print(gpcpd)
                        population = Master.iloc[m_index,pop_index]
                        if population < 1:
                            population = 1
                        water_produced = population*gpcpd*365*Water_diff
                    else:
                        water_produced = EAR_Water
                    if well_cap:
                        water_produced = 600000
                        # print('--worked')
                    if Master['Number of Sources'][m_index] < 1:
                        Master['Number of Sources'][m_index] = 1
                    source_water = water_produced/Master['Number of Sources'][m_index]
                    NEW_SET.iloc[new_i,num_sources_index] = Master['Number of Sources'][m_index]
                    NEW_SET.iloc[new_i,system_flow_index] = water_produced
                    NEW_SET.iloc[new_i,source_flow_index] = source_water

                    ## Arsenic Treatment? ##
                    skip_cap = False
                    for ws in range(0,len(Arsenic_Data)):
                        facility = source[-3::]
                        new_facs = []
                        
                        A_ID = Arsenic_Data.iloc[ws,Arsenic_Data.columns.get_loc('Water System ID')]
                        F_IDs = Arsenic_Data.iloc[ws,Arsenic_Data.columns.get_loc('Cr6 Sources')]
                        bks = []
                        for o,char in enumerate(F_IDs):
                            if char == ',':
                                bks.append(o)
                        if len(bks) > 0:
                            for bk in bks:
                                new_facs.append(F_IDs[0:bk])
                                new_facs.append(F_IDs[(bk+1)::])
                        if source_sys_no == A_ID[2::]:
                            if len(new_facs) > 1:
                                if np.float(facility) == np.float(new_facs[0]):
                                    skip_cap = True
                                elif np.float(facility) == np.float(new_facs[1]):
                                    skip_cap = True
                            elif len(new_facs) > 0:
                                if np.float(facility) == np.float(new_facs[0]):
                                    skip_cap = True
                            else:
                                if np.float(facility) == np.float(F_IDs):
                                    skip_cap = True

                    SBA_cap_costs,SBA_OM_costs,SBA_disposal_cost,SBA_resin_costs,SBA_BVs,SBA_energy_cost,SBA_chem_cost,SBA_labor_cost,SBA_design_f,SBA_avg_f_bypass_gpm = Hexavalent_Chromium_Treatment_Costs_SBA(
                        MCL,source_water,NEW_SET.iloc[new_i,ann_finding_index],Master.iloc[m_index,sc_index],Sulfate_avg,Nitrate_avg)

                    WBA_cap_costs,WBA_OM_costs,WBA_disposal_cost,WBA_resin_costs,WBA_energy_cost,WBA_HCl_cost,WBA_NaOH_cost,WBA_labor_cost,WBA_design_f,WBA_avg_f_bypass_gpm = Hexavalent_Chromium_Treatment_Costs_WBA(MCL,source_water,NEW_SET.iloc[new_i,ann_finding_index],Master.iloc[m_index,sc_index])
                    WBA_chem_cost = WBA_HCl_cost+WBA_NaOH_cost

                    RCF_cap_costs,RCF_OM_costs,RCF_disposal_cost,RCF_chem_cost,RCF_energy_cost,RCF_labor_cost,RCF_design_f,RCF_avg_f_bypass_gpm = Hexavalent_Chromium_Treatment_Costs_RCF(
                        MCL,source_water,NEW_SET.iloc[new_i,ann_finding_index],Master.iloc[m_index,sc_index])
                    RCF_resin_costs = 0

                    # if skip_cap == True:
                    #     print('--- THIS ONE ALREADY HAS ARESENIC TREATMENT ---')
                    #     SBA_cap_costs = 0

                    ## SBA
                    SBA_a_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_Cap')
                    SBA_b_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_OM')
                    SBA_disposal_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_OM-disp')
                    SBA_resin_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_OM-resin')
                    SBA_chem_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_OM-chem')
                    SBA_labor_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_OM-labor')
                    SBA_energy_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_OM-energy')
                    SBA_designf_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_designf_gpm')
                    SBA_new_flow_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_SBA_newflow_gpm')

                    NEW_SET.iloc[new_i,SBA_a_index] = SBA_cap_costs 
                    NEW_SET.iloc[new_i,SBA_b_index] = SBA_OM_costs
                    NEW_SET.iloc[new_i,SBA_disposal_index] = SBA_disposal_cost
                    NEW_SET.iloc[new_i,SBA_resin_index] = SBA_resin_costs
                    NEW_SET.iloc[new_i,SBA_chem_index] = SBA_chem_cost
                    NEW_SET.iloc[new_i,SBA_labor_index] = SBA_labor_cost
                    NEW_SET.iloc[new_i,SBA_energy_index] = SBA_energy_cost
                    NEW_SET.iloc[new_i,SBA_designf_index] = SBA_design_f                   
                    NEW_SET.iloc[new_i,SBA_new_flow_index] = SBA_avg_f_bypass_gpm

                    ## WBA
                    WBA_a_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_Cap')
                    WBA_b_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_OM')
                    WBA_disposal_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_OM-disp')
                    WBA_resin_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_OM-resin')
                    WBA_chemicals_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_OM-chem')
                    WBA_labor_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_OM-labor')
                    WBA_energy_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_OM-energy')
                    WBA_designf_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_designf_gpm')
                    WBA_new_flow_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_WBA_newflow_gpm')

                    NEW_SET.iloc[new_i,WBA_a_index] = WBA_cap_costs
                    NEW_SET.iloc[new_i,WBA_b_index] = WBA_OM_costs
                    NEW_SET.iloc[new_i,WBA_disposal_index] = WBA_disposal_cost
                    NEW_SET.iloc[new_i,WBA_resin_index] = WBA_resin_costs
                    NEW_SET.iloc[new_i,WBA_chemicals_index] = WBA_chem_cost
                    NEW_SET.iloc[new_i,WBA_labor_index] = WBA_labor_cost
                    NEW_SET.iloc[new_i,WBA_energy_index] = WBA_energy_cost
                    NEW_SET.iloc[new_i,WBA_designf_index] = WBA_design_f
                    NEW_SET.iloc[new_i,WBA_new_flow_index] = WBA_avg_f_bypass_gpm

                    ## RCF 
                    RCF_a_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_Cap')
                    RCF_b_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_OM')
                    RCF_disposal_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_OM-disp')
                    RCF_chemicals_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_OM-chem')
                    RCF_labor_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_OM-labor')
                    RCF_energy_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_OM-energy')
                    RCF_designf_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_designf_gpm')
                    RCF_new_flow_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_RCF_newflow_gpm')
                    
                    NEW_SET.iloc[new_i,RCF_a_index] = RCF_cap_costs
                    NEW_SET.iloc[new_i,RCF_b_index] = RCF_OM_costs
                    NEW_SET.iloc[new_i,RCF_disposal_index] = RCF_disposal_cost
                    NEW_SET.iloc[new_i,RCF_chemicals_index] = RCF_chem_cost
                    NEW_SET.iloc[new_i,RCF_labor_index] = RCF_labor_cost
                    NEW_SET.iloc[new_i,RCF_energy_index] = RCF_energy_cost
                    NEW_SET.iloc[new_i,RCF_designf_index] = RCF_design_f
                    NEW_SET.iloc[new_i,RCF_new_flow_index] = RCF_avg_f_bypass_gpm

                    Chosen_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Treatment_chosen')
                    Chosen_annualized_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Annualized_cost')
                    Chosen_Cap_tot_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Cap_Cost_chosen-tot')
                    Chosen_Cap_ann_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Cap_Cost_chosen-ann')
                    Chosen_OM_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_OM_Cost_chosen')
                    Chosen_disposal_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Chosen_disposal')
                    Chosen_resin_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Chosen_resin')
                    Chosen_energy_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Chosen_energy')
                    Chosen_chemicals_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Chosen_chem')
                    Chosen_labor_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Chosen_labor')
                    Chosen_designf_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Chosen_designf_MG')
                    Chosen_flow_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Chosen_flow_MG')
                    Ann_SBA_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Ann_SBA')
                    Ann_WBA_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Ann_WBA')
                    Ann_RCF_index = NEW_SET.columns.get_loc('MCL-'+MCL+'_Ann_RCF')

                    Ann_SBA = SBA_cap_costs*0.0944 + SBA_OM_costs
                    Ann_WBA = WBA_cap_costs*0.0944 + WBA_OM_costs
                    Ann_RCF = RCF_cap_costs*0.0944 + RCF_OM_costs
                    choose = 'None'
                    if Ann_RCF < Ann_SBA:
                        if Ann_RCF < Ann_WBA:
                            choose = 'RCF'
                        else:
                            if Ann_WBA == 0:
                                choose = 'RCF'
                            else:
                                choose = 'WBA'
                    else:
                        if Ann_SBA < Ann_WBA:
                            choose = 'SBA'
                        else:
                            if Ann_WBA == 0:
                                choose = 'SBA'
                            else:
                                choose = 'WBA'

                    NEW_SET.iloc[new_i,Ann_SBA_index] = Ann_SBA
                    NEW_SET.iloc[new_i,Ann_WBA_index] = Ann_WBA
                    NEW_SET.iloc[new_i,Ann_RCF_index] = Ann_RCF
                    NEW_SET.iloc[new_i,Chosen_index] = choose
                    NEW_SET.iloc[new_i,Chosen_annualized_index] = eval(choose+'_cap_costs')*0.0944 + eval(choose+'_OM_costs') + monitoring_cost
                    NEW_SET.iloc[new_i,Chosen_Cap_tot_index] = eval(choose+'_cap_costs')
                    NEW_SET.iloc[new_i,Chosen_Cap_ann_index] = eval(choose+'_cap_costs')*0.0944
                    NEW_SET.iloc[new_i,Chosen_OM_index] = eval(choose+'_OM_costs')
                    NEW_SET.iloc[new_i,Chosen_disposal_index] = eval(choose+'_disposal_cost')
                    NEW_SET.iloc[new_i,Chosen_resin_index] = eval(choose+'_resin_costs')
                    NEW_SET.iloc[new_i,Chosen_energy_index] = eval(choose+'_energy_cost')
                    NEW_SET.iloc[new_i,Chosen_chemicals_index] = eval(choose+'_chem_cost')
                    NEW_SET.iloc[new_i,Chosen_labor_index] = eval(choose+'_labor_cost')
                    NEW_SET.iloc[new_i,Chosen_designf_index] = eval(choose+'_design_f')*60*24*365/1000000 #MG/yr
                    NEW_SET.iloc[new_i,Chosen_flow_index] = eval(choose+'_avg_f_bypass_gpm')*60*24*365/1000000 #MG/yr

                    ## Health Benefit ##
                    Risk_col = NEW_SET.columns.get_loc('MCL-'+MCL+'_Risk')
                    Cancer_col = NEW_SET.columns.get_loc('MCL-'+MCL+'_Avoided_Cancer_Cases')
                    Solo_Pop = NEW_SET.iloc[new_i,NEW_SET.columns.get_loc('Population TINWSYS')]/NEW_SET.iloc[new_i,num_sources_index]
                    NEW_SET.iloc[new_i,Risk_col] = (NEW_SET.iloc[new_i,ann_finding_index]-int(MCL)) * (10**(-6)) / PHG
                    NEW_SET.iloc[new_i,Cancer_col] = Solo_Pop * NEW_SET.iloc[new_i,Risk_col]

                else:
                    ## Monitoring Costs w/o Treatment ##
                    if NEW_SET.iloc[new_i,source_type_index] == 'GW ':
                        monitoring_cost = sampling_cost*(1/3)
                    else:
                        monitoring_cost = (1)*sampling_cost
                    NEW_SET.iloc[new_i,monit_index] = monitoring_cost

        else:
            print('missing:',source)
    print('Number not found:',minus)
    ## Save Results ##
    NEW_SET.to_csv(results_folder+specific_results_folder+'OG+results_new-EQs_%s.csv'%(suffix))
    return NEW_SET

print(OG_Data[OG_Data['Water System']=='CA4400774'])
## Drop sources we don't want right now (emergency, standby, etc.) ##
if os.path.exists(data_folder+'CLEANED2 - '+file):
    OG_Data = pd.read_csv(data_folder+'CLEANED2 - '+file,header=0,encoding='cp437')
    # OG_Data = Drop_Unwanted_Sources(OG_Data)
else:
    OG_Data = Drop_Unwanted_Sources(OG_Data)
    OG_Data.to_csv(data_folder+'CLEANED2 - '+file)


## Large Function with Inset Functions to Get Treatment Costs ##
if Calc_Treatment:
    NEW_SET = Calculate_All_Treatment(Potential_MCLs,OG_Data,Master,suffix,all_the_columns,Finding_Type,DLR) # heavy lifting
else:
    NEW_SET = pd.read_csv(results_folder+specific_results_folder+'OG+results_new-EQs_%s.csv'%(suffix),header=0)

# Treated = NEW_SET[NEW_SET['Overall Finding'] > 1] # data set of only treated sources and their data (including costs)
Treated = NEW_SET

Terms = ['SCHOOL','INSTITUTION','INSTITUTION/AGRICULTURAL','INDUSTRIAL/AGRICULTURAL','MEDICAL FACILITY','HOTEL/MOTEL','RECREATION AREA','SECONDARY RESIDENCES']

## Sort Data ##
if Tabulate:

    CWS = Treated[Treated['FED Type']=='C   ']
    # print('CWS',len(CWS))
    CWS = CWS[CWS['NME']!='WHOLESALER (SELLS WATER)']
    print('CWS',len(CWS))
    NTNC = Treated[Treated['FED Type']=='NTNC']
    # print('NTNC',len(NTNC))
    NTNC = NTNC[NTNC['NME']!='WHOLESALER (SELLS WATER)']
    print('NTNC',len(NTNC))

    TNC = Treated[Treated['FED Type']=='NC  ']
    print('TNC',len(TNC))
    Wholesalers = Treated[Treated['NME']=='WHOLESALER (SELLS WATER)']
    print('Wholesalers',len(Wholesalers))

    for i in Potential_MCLs:
        Master['MCL_'+i] = 0
    print(Master.columns)
    Master.set_index(Master['Water System No.'],inplace=True)
    SC_index = Treated.columns.get_loc('Service Connections')
    pop_index = Treated.columns.get_loc('Population TINWSYS')
    num_sources = Treated.columns.get_loc('Number of Sources')
    source_gals = Treated.columns.get_loc('Source Flow (G)') # may not be treated flow

    try:
        system_num = Treated.columns.get_loc('Water System')
    except:
        system_num = Treated.columns.get_loc('Water System #')
    source_type = Treated.columns.get_loc('Water Type Code')
    
    # for dataset in [CWS,NTNC,TNC,Wholesalers]:
    # dataset = Wholesalers
    for Treated in [CWS,NTNC,TNC,Wholesalers]:
        # Treated = CWS
        if Treated.equals(CWS):
            index1 = SC_index
            breaks = [100,200,1000,5000,10000]
            key='CWS'
        elif Treated.equals(NTNC):
            index1 = pop_index
            breaks = [50,100,200,400,1000]
            key='NTNC'
        else:
            index1 = SC_index
            breaks = [1000000000,1000000001,1000000002,1000000003,1000000004]
            if Treated.equals(TNC):
                key='TNC'
            elif Treated.equals(Wholesalers):
                key='Whole'
        Final_Table = pd.DataFrame(columns=['A','B','C','D','E','F','Total'])#,index=final_table)
        for i in Potential_MCLs:
            globals()['SC_counted_'+str(i)] = []
            globals()['POP_counted_'+str(i)] = []

        for column in new_columns:
            if 'Cap_Cost_chosen-tot' in column:
                MCL = column[4:6]
                print('MCL:',MCL)
                col_index = Treated.columns.get_loc(column)
                OM_index = Treated.columns.get_loc('MCL-'+MCL+'_OM_Cost_chosen')
                OM_disp_index = Treated.columns.get_loc('MCL-'+MCL+'_Chosen_disposal')
                OM_resin_index = Treated.columns.get_loc('MCL-'+MCL+'_Chosen_resin')
                OM_energy_index = Treated.columns.get_loc('MCL-'+MCL+'_Chosen_energy')
                OM_chem_index = Treated.columns.get_loc('MCL-'+MCL+'_Chosen_chem')
                OM_labor_index = Treated.columns.get_loc('MCL-'+MCL+'_Chosen_labor')
                treated_flow_index = Treated.columns.get_loc('MCL-'+MCL+'_Chosen_designf_MG')
                monitoring_index = Treated.columns.get_loc('MCL-'+MCL+'_monitoring')
                water_type_index = Treated.columns.get_loc('Water Type Code')
                compliance_index = Treated.columns.get_loc('MCL-'+MCL+'_Compliance_Plan')
                Cancer_case_index = Treated.columns.get_loc('MCL-'+MCL+'_Avoided_Cancer_Cases')
                finding_index = Treated.columns.get_loc('Overall Finding')
                for i in ['A','B','C','D','E','F','G']:
                    globals()['to_treat_both_'+str(i)] = []
                    globals()['to_treat_GW_'+str(i)] = []
                    globals()['to_treat_SW_'+str(i)] = []
                    globals()['Cap_Costs_'+str(i)] = []
                    globals()['OM_Costs_'+str(i)] = []
                    # globals()['OM_base_Costs_'+str(i)] = []
                    globals()['OM_resin_Costs_'+str(i)] = []
                    globals()['OM_disp_Costs_'+str(i)] = []
                    globals()['OM_energy_Costs_'+str(i)] = []
                    globals()['OM_chem_Costs_'+str(i)] = []
                    globals()['OM_labor_Costs_'+str(i)] = []
                    # globals()['Resin_Amount_L_'+str(i)] = []
                    globals()['Pop_'+str(i)] = []
                    globals()['SC_Total_'+str(i)] = []
                    globals()['Total_Indiv_Costs_'+str(i)] = []
                    globals()['Million_Gallons_'+str(i)] = []
                    globals()['Systems_'+str(i)] = []
                    globals()['Monitoring_'+str(i)] = []
                    globals()['Cancer_Cases_Avoided_'+str(i)] = []

                to_treat_all = []
                not_found = []
                sys_list = []
                for x,sys_num in enumerate(Treated.iloc[:,system_num]):
                    if Treated.iloc[x,finding_index] > int(MCL): # check O&M column
                        if sys_num not in sys_list:
                            sys_list.append(sys_num)
                Treated_Sources_per_System = pd.DataFrame(0,index=sys_list,columns=['num sources'])
                for x,sys_num in enumerate(Treated.iloc[:,system_num]):
                    if Treated.iloc[x,finding_index] > int(MCL):
                        Treated_Sources_per_System.loc[sys_num,'num sources'] += 1

                ## Break up Cost Results into Bins ##
                for x in range(len(Treated)):
                    if round(Treated.iloc[x,finding_index],4) > int(MCL): 
                            sys_num = Treated.iloc[x,system_num]
                            PS_code = Treated.iloc[x,Treated.columns.get_loc('PS Code')]
                            if PS_code =='3610036-014':
                                continue
                            else:
                                pass
                            if Treated.iloc[x,num_sources] < 1:
                                Treated.iloc[x,num_sources] = 1 # start here
                            to_treat_all.append(Treated.iloc[x,Treated.columns.get_loc('PS Code')])
                            letter = 'N'
                            if Treated.iloc[x,index1] < breaks[0]:
                                letter = 'A'
                            elif breaks[0] <= Treated.iloc[x,index1] < breaks[1]:
                                letter = 'B'
                            elif breaks[1] <= Treated.iloc[x,index1] < breaks[2]:
                                letter = 'C'
                            elif breaks[2] <= Treated.iloc[x,index1] < breaks[3]:
                                letter = 'D'
                            elif breaks[3] <= Treated.iloc[x,index1] < breaks[4]:
                                letter = 'E'
                            elif Treated.iloc[x,index1] >= breaks[4]:
                                letter = 'F'
                            else:
                                print('--- ISSUE ---')
                                not_found.append(Treated.iloc[x,Treated.columns.get_loc('PS Code')])
                            eval('to_treat_both_'+letter).append(Treated['PS Code'])
                            if Treated.iloc[x,water_type_index] == 'GW ':
                                eval('to_treat_GW_'+letter).append(Treated['PS Code'])
                                # print('GW')
                            elif Treated.iloc[x,water_type_index] == 'SW ':
                                eval('to_treat_SW_'+letter).append(Treated['PS Code'])
                                # print('SW')
                            elif Treated.iloc[x,water_type_index] == 'GU ':
                                eval('to_treat_SW_'+letter).append(Treated['PS Code'])
                            else:
                                eval('to_treat_SW_'+letter).append(Treated['PS Code'])
                                print('-----ERROR-----',Treated.iloc[x,water_type_index])

                            eval('Cap_Costs_'+letter).append(Treated.iloc[x,col_index])
                            eval('OM_Costs_'+letter).append(Treated.iloc[x,OM_index])
                            eval('Monitoring_'+letter).append(Treated.iloc[x,monitoring_index])
                            eval('Total_Indiv_Costs_'+letter).append((Treated.iloc[x,col_index]*0.0944 + Treated.iloc[x,OM_index])/(Treated.iloc[x,index1] / Treated_Sources_per_System.loc[sys_num,'num sources']))
                            eval('OM_resin_Costs_'+letter).append(Treated.iloc[x,OM_resin_index])
                            eval('OM_disp_Costs_'+letter).append(Treated.iloc[x,OM_disp_index])
                            eval('OM_energy_Costs_'+letter).append(Treated.iloc[x,OM_energy_index])
                            eval('OM_chem_Costs_'+letter).append(Treated.iloc[x,OM_chem_index])
                            eval('OM_labor_Costs_'+letter).append(Treated.iloc[x,OM_labor_index])
                            eval('Cancer_Cases_Avoided_'+letter).append(Treated.iloc[x,Cancer_case_index])
                            
                            only_sys_num = float(sys_num[2::])
                            Master.loc[only_sys_num,'MCL_'+MCL] += (Treated.iloc[x,col_index]*0.0944 + Treated.iloc[x,OM_index] + Treated.iloc[x,monitoring_index] + Treated.iloc[x,compliance_index])/(Treated.iloc[x,index1])/12 # Capital: Treated.iloc[x,col_index]*0.0944

                            if Treated.iloc[x,system_num] not in eval('POP_counted_'+MCL):
                                eval('Pop_'+letter).append(Treated.iloc[x,pop_index])
                                eval('POP_counted_'+MCL).append(Treated.iloc[x,system_num])

                            if Treated.iloc[x,system_num] not in eval('SC_counted_'+MCL):
                                eval('SC_Total_'+letter).append(Treated.iloc[x,SC_index])#/ Treated.iloc[x,num_sources])
                                eval('SC_counted_'+MCL).append(Treated.iloc[x,system_num])

                            eval('Million_Gallons_'+letter).append(Treated.iloc[x,source_gals]/1000000) #converted to MG # including bypass flow
                            eval('Systems_'+letter).append(Treated.iloc[x,system_num])

                ## Combine Resuts in Final Table ##
                for letter in ['A','B','C','D','E','F']:
                    Final_Table.loc[column,letter] = sum(eval('Cap_Costs_'+letter))
                    Final_Table.loc['OM_'+MCL,letter] = sum(eval('OM_Costs_'+letter))
                    Final_Table.loc['OM-resin_'+MCL,letter] = sum(eval('OM_resin_Costs_'+letter))
                    Final_Table.loc['OM-disposal_'+MCL,letter] = sum(eval('OM_disp_Costs_'+letter))
                    Final_Table.loc['OM-energy_'+MCL,letter] = sum(eval('OM_energy_Costs_'+letter))
                    Final_Table.loc['OM-chem_'+MCL,letter] = sum(eval('OM_chem_Costs_'+letter))
                    Final_Table.loc['OM-labor_'+MCL,letter] = sum(eval('OM_labor_Costs_'+letter))

                    Final_Table.loc['No. of Sources_'+MCL,letter] = len(eval('to_treat_both_'+letter))
                    Final_Table.loc['No. of Sources_GW'+MCL,letter] = len(eval('to_treat_GW_'+letter))
                    Final_Table.loc['No. of Sources_SW'+MCL,letter] = len(eval('to_treat_SW_'+letter))
                    Final_Table.loc['No. of Systems_'+MCL,letter] = len(set(eval('Systems_'+letter)))
                    Final_Table.loc['SC_'+MCL,letter] = sum(eval('SC_Total_'+letter))
                    Final_Table.loc['Pop_'+MCL,letter] = sum(eval('Pop_'+letter))
                    Final_Table.loc['MG_Treated_'+MCL,letter] = sum(eval('Million_Gallons_'+letter))  
                    Final_Table.loc['Avoided_Cancer_Cases_'+MCL,letter] = sum(eval('Cancer_Cases_Avoided_'+letter))

                Final_Table.loc[column,'Total'] = sum(Cap_Costs_A) + sum(Cap_Costs_B) + sum(Cap_Costs_C) + sum(Cap_Costs_D) + sum(Cap_Costs_E) + sum(Cap_Costs_F)
                Final_Table.loc['OM_'+MCL,'Total'] = sum(OM_Costs_A) + sum(OM_Costs_B) + sum(OM_Costs_C) + sum(OM_Costs_D) + sum(OM_Costs_E) + sum(OM_Costs_F)
                Final_Table.loc['OM-resin_'+MCL,'Total'] = sum(OM_resin_Costs_A) + sum(OM_resin_Costs_B) + sum(OM_resin_Costs_C) + sum(OM_resin_Costs_D) + sum(OM_resin_Costs_E) + sum(OM_resin_Costs_F)
                Final_Table.loc['OM-disposal_'+MCL,'Total'] = sum(OM_disp_Costs_A) + sum(OM_disp_Costs_B) + sum(OM_disp_Costs_C) + sum(OM_disp_Costs_D) + sum(OM_disp_Costs_E) + sum(OM_disp_Costs_F)
                Final_Table.loc['OM-energy_'+MCL,'Total'] = sum(OM_energy_Costs_A) + sum(OM_energy_Costs_B) + sum(OM_energy_Costs_C) + sum(OM_energy_Costs_D) + sum(OM_energy_Costs_E) + sum(OM_energy_Costs_F)
                Final_Table.loc['OM-chem_'+MCL,'Total'] = sum(OM_chem_Costs_A) + sum(OM_chem_Costs_B) + sum(OM_chem_Costs_C) + sum(OM_chem_Costs_D) + sum(OM_chem_Costs_E) + sum(OM_chem_Costs_F)
                Final_Table.loc['OM-labor_'+MCL,'Total'] = sum(OM_labor_Costs_A) + sum(OM_labor_Costs_B) + sum(OM_labor_Costs_C) + sum(OM_labor_Costs_D) + sum(OM_labor_Costs_E) + sum(OM_labor_Costs_F)

                Final_Table.loc['No. of Sources_'+MCL,'Total'] = len(to_treat_all)
                Final_Table.loc['No. of Sources_GW'+MCL,'Total'] = len(to_treat_GW_A) + len(to_treat_GW_B) + len(to_treat_GW_C) + len(to_treat_GW_D) + len(to_treat_GW_E) + len(to_treat_GW_F)
                Final_Table.loc['No. of Sources_SW'+MCL,'Total'] = len(to_treat_SW_A) + len(to_treat_SW_B) + len(to_treat_SW_C) + len(to_treat_SW_D) + len(to_treat_SW_E) + len(to_treat_SW_F)
                Final_Table.loc['No. of Systems_'+MCL,'Total'] = len(set(Systems_A)) + len(set(Systems_B)) + len(set(Systems_C)) + len(set(Systems_D)) + len(set(Systems_E)) + len(set(Systems_F))
                Final_Table.loc['Pop_'+MCL,'Total'] = sum(Pop_A) + sum(Pop_B) + sum(Pop_C) + sum(Pop_D) + sum(Pop_E) + sum(Pop_F)
                Final_Table.loc['SC_'+MCL,'Total'] = sum(SC_Total_A) + sum(SC_Total_B) + sum(SC_Total_C) + sum(SC_Total_D) + sum(SC_Total_E) + sum(SC_Total_F)
                Final_Table.loc['MG_Treated_'+MCL,'Total'] = sum(Million_Gallons_A) + sum(Million_Gallons_B) + sum(Million_Gallons_C) + sum(Million_Gallons_D) + sum(Million_Gallons_E) + sum(Million_Gallons_F)
                Final_Table.loc['Avoided_Cancer_Cases_'+MCL,'Total'] = sum(Cancer_Cases_Avoided_A) + sum(Cancer_Cases_Avoided_B) + sum(Cancer_Cases_Avoided_C) + sum(Cancer_Cases_Avoided_D) + sum(Cancer_Cases_Avoided_E) + sum(Cancer_Cases_Avoided_F)
        
        print(Final_Table)
        Final_Table = Final_Table.sort_index(axis=0,ascending=True)
        Final_Table.to_csv(results_folder+specific_results_folder+'Final_Table_%s.csv'%(suffix+'_for-'+key))
        if key == 'CWS':
            Master.to_csv(results_folder+specific_results_folder+'Monthly_Cost_per_Connection_incl-monitoring_%s_%s.csv'%(key,suffix))

