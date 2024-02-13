from pytest import approx
import unittest
from green_steel_ammonia_define_scenarios import run_scenarios, set_up_scenarios

def get_settings(atb_year=0, policy=0, site=0, electrolysis_case=0, electrolysis_cost_case=0, grid_connection_case=0, storage_capacity_case=0):

    atb_years = [
                2020,
                2025,
                2030,
                2035
                ][atb_year]

    if policy == 0: policy='no-policy'
    elif policy == 1: policy = "base"
    elif policy == 2: policy = "max"
    policy = {policy: {
        'no-policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0, 'Storage ITC': 0},
        'base': {'Wind ITC': 0, 'Wind PTC':  0.0055341, "H2 PTC": 0.6, 'Storage ITC': 0.06},
        'max': {'Wind ITC': 0, 'Wind PTC': 0.0332046, "H2 PTC": 3.0, 'Storage ITC': 0.5},
    }[policy]}
    


    site_selection = [
                    'Site 1', # not working
                    'Site 2',
                    'Site 3',
                    'Site 4',
                    'Site 5'
                    ][site]

    electrolysis_cases = [
                          'Centralized',
                          'Distributed'
                          ][electrolysis_case]

    electrolyzer_cost_cases = [
                                'Low',
                                'Mid',
                                'High'
                                ][electrolysis_cost_case]

    grid_connection_cases = [
                            'off-grid',
                            'grid-only', # not working 
                            'hybrid-grid' # not working
                            ][grid_connection_case]

    storage_capacity_cases = [
                            1.0,
                            1.25,
                            1.5
                            ][storage_capacity_case]

    num_pem_stacks= 6 # Doesn't actually do anything
    run_solar_param_sweep=False

    return [atb_years], policy, [site_selection], [electrolysis_cases], [electrolyzer_cost_cases], [grid_connection_cases], [storage_capacity_cases], num_pem_stacks, run_solar_param_sweep

class TestGreenSteelExampleSite5NoPolicyCentralizedMidOffGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleSite5NoPolicyCentralizedMidOffGrid, self).setUpClass()

        atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep = get_settings(2, 0, 4, 0, 1, 0, 0)
        
        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep)
        
        self.scenario = {}
        self.policy_option, self.turbine_model, self.scenario['Useful Life'], self.wind_cost_kw, self.solar_cost_kw,\
        self.scenario['Debt Equity'], self.atb_year, self.scenario['H2 PTC'], self.scenario['Wind ITC'],\
        self.discount_rate, self.tlcc_wind_costs, self.tlcc_solar_costs, self.tlcc_hvdc_costs, self.tlcc_total_costs, self.run_RODeO_selector, self.lcoh,\
        self.wind_itc_total, self.total_itc_hvdc, self.hopp_dict = run_scenarios(arg_list)

    def test_lcoh(self):
        assert self.lcoh == approx(4.3012984737802595)

    def test_tlcc_wind_costs(self):
        assert self.tlcc_wind_costs == approx(-2468620682.4134855)

    def test_tlcc_solar_costs(self):
        assert self.tlcc_solar_costs == approx(0)

    def test_tlcc_hvdc_costs(self):
        assert self.tlcc_hvdc_costs == approx(0)
    
    def test_tlcc_total_costs(self):
        assert self.tlcc_total_costs == approx(-2468620682.4134855)

    def test_wind_itc_total(self):
        assert self.wind_itc_total == approx(0)

    def test_steel_breakeven_price(self):
        assert self.hopp_dict.main_dict["Models"]['steel_LCOS']['output_dict']["steel_breakeven_price"] == approx(961.4620566701477)

class TestGreenSteelExampleSite1NoPolicyCentralizedMidOffGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleSite1NoPolicyCentralizedMidOffGrid, self).setUpClass()

        atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep = get_settings(2, 0, 0, 0, 1, 0, 0)

        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, electrolyzer_cost_cases, grid_connection_cases, \
        storage_capacity_cases, num_pem_stacks, run_solar_param_sweep)
        
        self.scenario = {}
        self.policy_option, self.turbine_model, self.scenario['Useful Life'], self.wind_cost_kw, self.solar_cost_kw,\
        self.scenario['Debt Equity'], self.atb_year, self.scenario['H2 PTC'], self.scenario['Wind ITC'],\
        self.discount_rate, self.tlcc_wind_costs, self.tlcc_solar_costs, self.tlcc_hvdc_costs, self.tlcc_total_costs, self.run_RODeO_selector, self.lcoh,\
        self.wind_itc_total, self.total_itc_hvdc, self.hopp_dict = run_scenarios(arg_list)

    def test_lcoh(self):
        assert self.lcoh == approx(10.934350707262357)

    def test_tlcc_wind_costs(self):
        assert self.tlcc_wind_costs == approx(-2541253831.601611)

    def test_tlcc_solar_costs(self):
        assert self.tlcc_solar_costs == approx(0)

    def test_tlcc_hvdc_costs(self):
        assert self.tlcc_hvdc_costs == approx(0)
    
    def test_tlcc_total_costs(self):
        assert self.tlcc_total_costs == approx(-2541253831.601611)

    def test_wind_itc_total(self):
        assert self.wind_itc_total == approx(0)

    def test_steel_breakeven_price(self):
        assert self.hopp_dict.main_dict["Models"]['steel_LCOS']['output_dict']["steel_breakeven_price"] == approx(1442.4015958776993)

class TestGreenSteelExampleSite2BasePolicyCentralizedMidOffGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleSite2BasePolicyCentralizedMidOffGrid, self).setUpClass()

        atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep = get_settings(2, 1, 1, 0, 1, 0, 0)

        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, electrolyzer_cost_cases, grid_connection_cases, \
        storage_capacity_cases, num_pem_stacks, run_solar_param_sweep)
        
        self.scenario = {}
        self.policy_option, self.turbine_model, self.scenario['Useful Life'], self.wind_cost_kw, self.solar_cost_kw,\
        self.scenario['Debt Equity'], self.atb_year, self.scenario['H2 PTC'], self.scenario['Wind ITC'],\
        self.discount_rate, self.tlcc_wind_costs, self.tlcc_solar_costs, self.tlcc_hvdc_costs, self.tlcc_total_costs, self.run_RODeO_selector, self.lcoh,\
        self.wind_itc_total, self.total_itc_hvdc, self.hopp_dict = run_scenarios(arg_list)

    def test_lcoh(self):
        assert self.lcoh == approx(3.225650628339138)

    def test_tlcc_wind_costs(self):
        assert self.tlcc_wind_costs == approx(-1716551702.4435363)

    def test_tlcc_solar_costs(self):
        assert self.tlcc_solar_costs == approx(0)

    def test_tlcc_hvdc_costs(self):
        assert self.tlcc_hvdc_costs == approx(0)
    
    def test_tlcc_total_costs(self):
        assert self.tlcc_total_costs == approx(-1716551702.4435363)

    def test_wind_itc_total(self):
        assert self.wind_itc_total == approx(0)

    def test_steel_breakeven_price(self):
        assert self.hopp_dict.main_dict["Models"]['steel_LCOS']['output_dict']["steel_breakeven_price"] == approx(1014.4970213894144)

class TestGreenSteelExampleSite2BasePolicyDistributedMidOffGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleSite2BasePolicyDistributedMidOffGrid, self).setUpClass()

        atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep = get_settings(2, 1, 1, 1, 1, 0, 0)

        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, electrolyzer_cost_cases, grid_connection_cases, \
        storage_capacity_cases, num_pem_stacks, run_solar_param_sweep)
        
        self.scenario = {}
        self.policy_option, self.turbine_model, self.scenario['Useful Life'], self.wind_cost_kw, self.solar_cost_kw,\
        self.scenario['Debt Equity'], self.atb_year, self.scenario['H2 PTC'], self.scenario['Wind ITC'],\
        self.discount_rate, self.tlcc_wind_costs, self.tlcc_solar_costs, self.tlcc_hvdc_costs, self.tlcc_total_costs, self.run_RODeO_selector, self.lcoh,\
        self.wind_itc_total, self.total_itc_hvdc, self.hopp_dict = run_scenarios(arg_list)

    def test_lcoh(self):
        assert self.lcoh == approx(4.0223827775471905)

    def test_tlcc_wind_costs(self):
        assert self.tlcc_wind_costs == approx(-1716551702.4435363)

    def test_tlcc_solar_costs(self):
        assert self.tlcc_solar_costs == approx(0)

    def test_tlcc_hvdc_costs(self):
        assert self.tlcc_hvdc_costs == approx(0)
    
    def test_tlcc_total_costs(self):
        assert self.tlcc_total_costs == approx(-1716551702.4435363)

    def test_wind_itc_total(self):
        assert self.wind_itc_total == approx(0)

    def test_steel_breakeven_price(self):
        assert self.hopp_dict.main_dict["Models"]['steel_LCOS']['output_dict']["steel_breakeven_price"] == approx(1068.387984000575)

class TestGreenSteelExampleSite2BasePolicyDistributedHighOffGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleSite2BasePolicyDistributedHighOffGrid, self).setUpClass()

        atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep = get_settings(2, 1, 1, 1, 2, 0, 0)

        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, electrolyzer_cost_cases, grid_connection_cases, \
        storage_capacity_cases, num_pem_stacks, run_solar_param_sweep)
        
        self.scenario = {}
        self.policy_option, self.turbine_model, self.scenario['Useful Life'], self.wind_cost_kw, self.solar_cost_kw,\
        self.scenario['Debt Equity'], self.atb_year, self.scenario['H2 PTC'], self.scenario['Wind ITC'],\
        self.discount_rate, self.tlcc_wind_costs, self.tlcc_solar_costs, self.tlcc_hvdc_costs, self.tlcc_total_costs, self.run_RODeO_selector, self.lcoh,\
        self.wind_itc_total, self.total_itc_hvdc, self.hopp_dict = run_scenarios(arg_list)

    def test_lcoh(self):
        assert self.lcoh == approx(4.4764088847829075)

    def test_tlcc_wind_costs(self):
        assert self.tlcc_wind_costs == approx(-1716551702.4435363)

    def test_tlcc_solar_costs(self):
        assert self.tlcc_solar_costs == approx(0)

    def test_tlcc_hvdc_costs(self):
        assert self.tlcc_hvdc_costs == approx(0)
    
    def test_tlcc_total_costs(self):
        assert self.tlcc_total_costs == approx(-1716551702.4435363)

    def test_wind_itc_total(self):
        assert self.wind_itc_total == approx(0)

    def test_steel_breakeven_price(self):
        assert self.hopp_dict.main_dict["Models"]['steel_LCOS']['output_dict']["steel_breakeven_price"] == approx(1119.4232080433294)

class TestGreenSteelExampleSite2MaxPolicyDistributedHighOffGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleSite2MaxPolicyDistributedHighOffGrid, self).setUpClass()

        atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep = get_settings(2, 2, 1, 1, 2, 0, 0)

        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, electrolyzer_cost_cases, grid_connection_cases, \
        storage_capacity_cases, num_pem_stacks, run_solar_param_sweep)
        
        self.scenario = {}
        self.policy_option, self.turbine_model, self.scenario['Useful Life'], self.wind_cost_kw, self.solar_cost_kw,\
        self.scenario['Debt Equity'], self.atb_year, self.scenario['H2 PTC'], self.scenario['Wind ITC'],\
        self.discount_rate, self.tlcc_wind_costs, self.tlcc_solar_costs, self.tlcc_hvdc_costs, self.tlcc_total_costs, self.run_RODeO_selector, self.lcoh,\
        self.wind_itc_total, self.total_itc_hvdc, self.hopp_dict = run_scenarios(arg_list)

    def test_lcoh(self):
        assert self.lcoh == approx(0.7235096102503652)

    def test_tlcc_wind_costs(self):
        assert self.tlcc_wind_costs == approx(-992212460.6479578)

    def test_tlcc_solar_costs(self):
        assert self.tlcc_solar_costs == approx(0)

    def test_tlcc_hvdc_costs(self):
        assert self.tlcc_hvdc_costs == approx(0)
    
    def test_tlcc_total_costs(self):
        assert self.tlcc_total_costs == approx(-992212460.6479578)

    def test_wind_itc_total(self):
        assert self.wind_itc_total == approx(0)

    def test_steel_breakeven_price(self):
        assert self.hopp_dict.main_dict["Models"]['steel_LCOS']['output_dict']["steel_breakeven_price"] == approx(869.1639874034265)

class TestGreenSteelExampleSite2MaxPolicyDistributedHighOnGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleSite2MaxPolicyDistributedHighOnGrid, self).setUpClass()

        atb_years, policy, site_selection, electrolysis_cases, \
            electrolyzer_cost_cases, grid_connection_cases, \
                storage_capacity_cases, num_pem_stacks, run_solar_param_sweep = get_settings(2, 2, 1, 1, 2, 1, 0)

        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, electrolyzer_cost_cases, grid_connection_cases, \
        storage_capacity_cases, num_pem_stacks, run_solar_param_sweep)
        
        self.scenario = {}
        self.policy_option, self.turbine_model, self.scenario['Useful Life'], self.wind_cost_kw, self.solar_cost_kw,\
        self.scenario['Debt Equity'], self.atb_year, self.scenario['H2 PTC'], self.scenario['Wind ITC'],\
        self.discount_rate, self.tlcc_wind_costs, self.tlcc_solar_costs, self.tlcc_hvdc_costs, self.tlcc_total_costs, self.run_RODeO_selector, self.lcoh,\
        self.wind_itc_total, self.total_itc_hvdc, self.hopp_dict = run_scenarios(arg_list)

    def test_lcoh(self):
        assert self.lcoh == approx(0.7235096102503652)

    def test_tlcc_wind_costs(self):
        assert self.tlcc_wind_costs == approx(-992212460.6479578)

    def test_tlcc_solar_costs(self):
        assert self.tlcc_solar_costs == approx(0)

    def test_tlcc_hvdc_costs(self):
        assert self.tlcc_hvdc_costs == approx(0)
    
    def test_tlcc_total_costs(self):
        assert self.tlcc_total_costs == approx(-992212460.6479578)

    def test_wind_itc_total(self):
        assert self.wind_itc_total == approx(0)

    def test_steel_breakeven_price(self):
        assert self.hopp_dict.main_dict["Models"]['steel_LCOS']['output_dict']["steel_breakeven_price"] == approx(869.1639874034265)
