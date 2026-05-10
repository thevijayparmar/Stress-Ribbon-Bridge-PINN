import math
import csv
import itertools

# =============================================================================
# 1. CORE CATENARY MATH FUNCTIONS 
# =============================================================================
def _safe_a_min(L: float) -> float:
    return max(L / 700.0, 1e-3)

def _x0(L: float, dz: float, a: float) -> float:
    """Calculates horizontal distance from left support to vertex."""
    S = math.sinh(L / (2 * a))
    return L / 2 - a * math.asinh(dz / (2 * a * S))

def _sag_pair(L: float, dz: float, a: float):
    """Calculates vertical drop from each support to the vertex."""
    x0 = _x0(L, dz, a)
    cosh0 = math.cosh(x0 / a)
    return a * (cosh0 - 1), a * (math.cosh((L - x0) / a) - 1)

def _tension_pair(L: float, dz: float, a: float, w: float):
    """Calculates tension at supports and identifies T_max."""
    x0 = _x0(L, dz, a)
    H  = w * a
    V1, V2 = w * (0 - x0), w * (L - x0)
    T1, T2 = math.hypot(H, V1), math.hypot(H, V2)
    return T1, T2, max(T1, T2)

def _a_min_Tmax(L, dz, w) -> float:
    """Finds 'a' parameter for the absolute minimum possible tension."""
    a_low, a_high = _safe_a_min(L), 1e6
    g = 0.5 * (math.sqrt(5) - 1)
    while abs(a_high - a_low) > 1e-6 * (a_low + a_high) * 0.5:
        a1 = a_low + (1 - g) * (a_high - a_low)
        a2 = a_low + g * (a_high - a_low)
        if _tension_pair(L, dz, a1, w)[2] < _tension_pair(L, dz, a2, w)[2]:
            a_high = a2
        else:
            a_low = a1
    return 0.5 * (a_high + a_low)

def _a_for_Tmax(L, dz, w, T_target) -> float:
    """Finds 'a' parameter to match a specific target tension (Tallow)."""
    a_low = _a_min_Tmax(L, dz, w)
    if _tension_pair(L, dz, a_low, w)[2] >= T_target:
        return a_low
    # Incremental search to find upper bound (Solving the Ghost NaN issue)
    a_high = a_low * 2
    while _tension_pair(L, dz, a_high, w)[2] < T_target and a_high < 1e9:
        a_high *= 2
    # Bisection to find exact 'a'
    while abs(a_high - a_low) > 1e-6 * (a_high + a_low) * 0.5:
        a_mid = 0.5 * (a_low + a_high)
        if _tension_pair(L, dz, a_mid, w)[2] < T_target:
            a_low = a_mid
        else:
            a_high = a_mid
    return 0.5 * (a_low + a_high)

def _a_for_sag(L, dz, w, sag_target, a_max_search=1e8):
    """Finds 'a' parameter to match a requested target sag."""
    # Check if sag is mathematically possible (Target Sag must be >= dz)
    if sag_target < dz - 1e-4:
        return None
        
    a_high = a_max_search
    # Check if target is achievable at 'infinite' tension (Straight diagonal line)
    # Using vertex-based sag for consistency with dataset definition
    sags_at_high = _sag_pair(L, dz, a_high)
    if max(sags_at_high) > sag_target:
        return None
        
    a_low = _safe_a_min(L)
    while max(_sag_pair(L, dz, a_low)) <= sag_target:
        a_low *= 0.5
        if a_low < 1e-6: break # Prevent infinite loops
        
    while abs(a_high - a_low) > 1e-6 * (a_high + a_low) * 0.5:
        a_mid = 0.5 * (a_low + a_high)
        if max(_sag_pair(L, dz, a_mid)) > sag_target:
            a_low = a_mid
        else:
            a_high = a_mid
    return a_high

# =============================================================================
# 2. GENERATOR LOGIC
# =============================================================================
def generate_bridge_data():
    spans = [2, 5, 10, 20, 40, 50, 80, 100, 120, 125, 150, 175, 200, 250, 300, 500, 750, 1000, 1500, 1800, 2000]
    height_diffs = [0, 0.5, 1, 2, 6, 8, 10, 12, 15, 20, 30, 40, 45, 50]
    udls = [0.5, 1, 5, 10, 15, 20, 25, 30, 40, 50, 100, 200]
    t_allows = [5, 10, 50, 100, 300, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 100000, 150000, 200000, 500000, 750000, 900000, 1000000]
    sag_percents = [0.25, 2.5, 5, 10, 50, 75, 100, 150, 200, 300, 400, 500]

    headers = [
        "In_Span_m", "In_HeightDiff_m", "In_DirectLength_m", "In_UDL_kNm", "In_Tallow_kN", "In_RecSag_m", "In_RecSag_Pt_pct",
        "Out_Param_a_m", "Out_CableLength_m", "Out_SagAchieved", "Out_ControlMode",
        "Out_SagStart_m", "Out_SagEnd_m", "Out_AbsMaxSag_m", 
        "Out_SlopeStart_deg", "Out_SlopeEnd_deg", 
        "Out_TensionStart_kN", "Out_TensionEnd_kN", 
        "Out_V1_kN", "Out_V2_kN", "Out_H_kN", 
        "Out_Utilization", "Out_Tmin_kN", "Out_a_minT_m", 
        "Out_Vertex_x0_m", "Out_Req_Tallow_kN", "Out_Geometry_Possible"
    ]

    combinations = list(itertools.product(spans, height_diffs, udls, t_allows, sag_percents))
    total_combs = len(combinations)
    print(f"Total possible combinations to evaluate: {total_combs}")

    chunk_size = 1000000
    file_index = 1
    current_chunk_data = []

    for i, (L, dz, w, Tallow, sag_pct) in enumerate(combinations):
        SagRec = L * (sag_pct / 100.0)
        direct_len = math.hypot(L, dz)

        # 1. Baseline Feasibility (Absolute Min Tension to just hang the cable)
        a_minT = _a_min_Tmax(L, dz, w)
        T_min = _tension_pair(L, dz, a_minT, w)[2]
        geometry_possible = 1 if Tallow >= (T_min - 1e-6) else 0

        if geometry_possible == 1:
            # 2. DESIGN-FIRST PRIORITY: Solve for Target Sag first
            # We search for 'a' that gives SagRec without initially capping by Tallow
            a_target_sag = _a_for_sag(L, dz, w, SagRec, a_max_search=1e7)
            
            T_req = None
            if a_target_sag is not None:
                T_req = _tension_pair(L, dz, a_target_sag, w)[2]

            # 3. Decision Logic: Sag-Controlled vs Capacity-Controlled
            if T_req is not None and T_req <= Tallow:
                # [cite_start]Engineering Intent Achieved: We only tight the cable as much as needed [cite: 20]
                a_opt = a_target_sag
                Control_Mode = 1 # Sag-controlled
                Req_Tallow = round(T_req, 2)
            else:
                # [cite_start]Capacity Limited: Pull the cable to its allowable limit [cite: 31]
                a_opt = _a_for_Tmax(L, dz, w, Tallow)
                Control_Mode = 0 # Capacity-controlled
                # If sag was mathematically possible, Req_Tallow is that tension, otherwise NaN
                Req_Tallow = round(T_req, 2) if T_req is not None else "NaN"

            # 4. Extract Output Variables
            T1, T2, T_max = _tension_pair(L, dz, a_opt, w)
            sag1, sag2 = _sag_pair(L, dz, a_opt)
            x0 = _x0(L, dz, a_opt)
            
            # [cite_start]GEOMETRIC GUARDRAIL: If vertex is off-span, Absolute Sag is just the height difference [cite: 33]
            if x0 < 0 or x0 > L:
                abs_max_sag = dz 
            else:
                abs_max_sag = max(sag1, sag2)
                
            cable_len = a_opt * (math.sinh((L - x0)/a_opt) + math.sinh(x0/a_opt))
            Sag_Achieved = 1 if abs_max_sag <= (SagRec + 1e-4) else 0
            
            slope1 = abs(math.degrees(math.atan(math.sinh(-x0 / a_opt))))
            slope2 = abs(math.degrees(math.atan(math.sinh((L - x0) / a_opt))))
            V1 = abs(w * (0 - x0))
            V2 = abs(w * (L - x0))
            H = w * a_opt
            Utilization = T_max / Tallow

            row = [
                L, dz, round(direct_len, 4), w, Tallow, round(SagRec, 4), sag_pct,
                round(a_opt, 4), round(cable_len, 4), Sag_Achieved, Control_Mode,
                round(sag1, 4), round(sag2, 4), round(abs_max_sag, 4),
                round(slope1, 2), round(slope2, 2),
                round(T1, 2), round(T2, 2),
                round(V1, 2), round(V2, 2), round(H, 2),
                round(Utilization, 4), round(T_min, 2), round(a_minT, 4),
                round(x0, 4), Req_Tallow, geometry_possible
            ]
        else:
            # [cite_start]Physically impossible geometries [cite: 37]
            row = [L, dz, round(direct_len, 4), w, Tallow, round(SagRec, 4), sag_pct] + ["NaN"] * 15 + [round(T_min, 2), round(a_minT, 4), "NaN", "NaN", geometry_possible]
        
        current_chunk_data.append(row)

        if len(current_chunk_data) == chunk_size:
            filename = f"Bridge_AI_Data_Part_{file_index}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(current_chunk_data)
            print(f"Saved {filename} ({chunk_size} rows)")
            file_index += 1
            current_chunk_data = []

    if current_chunk_data:
        filename = f"Bridge_AI_Data_Part_{file_index}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(current_chunk_data)
        print(f"Saved {filename} ({len(current_chunk_data)} rows)")

    print(f"\n✅ Generation Complete! Processed {total_combs} combinations.")

if __name__ == "__main__":
    generate_bridge_data()