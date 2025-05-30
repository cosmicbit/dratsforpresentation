

def generate_routefile(choices):

    with open("oneway/one_intersection/routes3.rou.xml", "w") as routes:
        print("""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Define a vehicle type -->
    <vType id="car" accel="1.0" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.9"/>

    <!-- Define routes through the intersection -->
    <!-- Example Route 1: Vehicles coming from North and exiting to East -->
    <route id="N_to_E" edges="edge_n_in edge_e_out"/>
    <route id="N_to_S" edges="edge_n_in edge_s_out"/>
    <route id="N_to_W" edges="edge_n_in edge_w_out"/>

    <route id="W_to_N" edges="edge_w_in edge_n_out"/>
    <route id="W_to_E" edges="edge_w_in edge_e_out"/>
    <route id="W_to_S" edges="edge_w_in edge_s_out"/>

    <route id="E_to_N" edges="edge_e_in edge_n_out"/>
    <route id="E_to_S" edges="edge_e_in edge_s_out"/>
    <route id="E_to_W" edges="edge_e_in edge_w_out"/>

    <route id="S_to_N" edges="edge_s_in edge_n_out"/>
    <route id="S_to_E" edges="edge_s_in edge_e_out"/>
    <route id="S_to_W" edges="edge_s_in edge_w_out"/>""", file=routes)

        if choices[0] == 1:
            print("""
            <flow id="flow1" type="car" route="N_to_E" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow2" type="car" route="N_to_S" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow3" type="car" route="N_to_W" begin="0" end="1000" vehsPerHour="600"/>
            """, file=routes)
        if choices[1] == 1:
            print("""
            <flow id="flow7" type="car" route="E_to_N" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow8" type="car" route="E_to_S" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow9" type="car" route="E_to_W" begin="0" end="1000" vehsPerHour="600"/>
            """, file=routes)
        if choices[2] == 1:
            print("""
            <flow id="flow10" type="car" route="S_to_N" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow11" type="car" route="S_to_E" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow12" type="car" route="S_to_W" begin="0" end="1000" vehsPerHour="600"/>
            
            """, file=routes)
        if choices[3] == 1:
            print("""
            
            <flow id="flow4" type="car" route="W_to_N" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow5" type="car" route="W_to_E" begin="0" end="1000" vehsPerHour="600"/>
            <flow id="flow6" type="car" route="W_to_S" begin="0" end="1000" vehsPerHour="600"/>
            """, file=routes)
        print("</routes>", file=routes)
