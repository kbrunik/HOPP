from pytest import approx
import unittest
from green_steel_ammonia_define_scenarios import run_scenarios, set_up_scenarios

class TestGreenSteelExampleMNNoPolicyOffGrid(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleMNNoPolicyOffGrid, self).setUpClass()

        atb_years = [2030]

        policy = {'no-policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0, 'Storage ITC': 0}}

        site_selection = ['Site 5']

        electrolysis_cases = ['Centralized']

        electrolyzer_cost_cases = ['Mid']

        grid_connection_cases = ['off-grid'   ]

        storage_capacity_cases = [1.0]

        num_pem_stacks= 6 # Doesn't actually do anything
        run_solar_param_sweep=False

        arg_list = set_up_scenarios(atb_years, policy, site_selection, electrolysis_cases, electrolyzer_cost_cases, grid_connection_cases, \
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

class TestGreenSteelExampleMNNoPolicyOffGrid2(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestGreenSteelExampleMNNoPolicyOffGrid2, self).setUpClass()

        atb_years = [2030]

        policy = {'no-policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0, 'Storage ITC': 0}}

        site_selection = ['Site 1']

        electrolysis_cases = ['Centralized']

        electrolyzer_cost_cases = ['Mid']

        grid_connection_cases = ['off-grid']

        storage_capacity_cases = [1.0]

        num_pem_stacks= 6 # Doesn't actually do anything
        run_solar_param_sweep=False

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
