import numpy as np 
from pulp import *
import pandas as pd

class tsp_drone:
  '''
  class instance initiaton
  '''
  def __init__(self):
        None

  '''
  Exact formulation of TSP
  '''
  def tsp_drone(self, road_matrix,drone_factor = 1):
    result = []
    result_name = []
    result_df = pd.DataFrame()
    
    drone_matrix = road_matrix*drone_factor
    row,col = road_matrix.shape
 
    M=1000

    problem = LpProblem('TravellingSalesmanProblem', LpMinimize)

    # Decision variable X for truck route
    decisionVariableX = LpVariable.dicts('decisionVariable_X', ((i, j) \
                                                                for i in range(row) for j in range(row)),
                                        lowBound=0, upBound=1, cat='Integer')
    # Decision variable Y for Drone route
    decisionVariableY = LpVariable.dicts('decisionVariable_Y', ((i, j, k) \
                                                                for i in range(row) for j in range(row) for k in range(row)), lowBound=0, upBound=1, cat='Integer')

    # Decision variable D for drone arrival time
    decisionVariableD = LpVariable.dicts('Drone_time', (i for i in range(row)), lowBound=0, cat = 'Float')

    # subtours elimination truck
    u = LpVariable.dicts('u', (i for i in range(row)), lowBound=1, cat='Integer')
    
    
    # subtours elimination drones
    v = LpVariable.dicts('v', (i for i in range(row)), lowBound=0, cat='Integer')

    # pij variable
    p = LpVariable.dicts("drone_position", ((i,j) for i in range(row) for j in range(row)), lowBound=0, upBound=1, cat='Integer')
    
    # objective variable
    z = LpVariable("Objective_z", lowBound=0, cat='Float')

    #  Decision variable T for truck arrival time
    decisionVariableT = LpVariable.dict('Time_truck',(i for i in range(row)), lowBound=0, cat='Float')

    # Objective Function
    problem += lpSum(road_matrix.iloc[i,j] * decisionVariableX[i, j] for i in range(row) for j in range(row))
    
    # Constraints
    for i in range(row):
      problem += (decisionVariableX[i,i] == 0) # elimination of (1 to 1) route
      problem += (decisionVariableY[i,i,i] == 0) # elimination of (1 to 1 to 1) route for drone
      if (i==0):
        problem += lpSum(decisionVariableX[i,j] for j in range(row))==1 # truck leaves the depot once
        problem += lpSum(decisionVariableX[j,i] for j in range(row)) ==1 #truck reaches the depot once
        problem += lpSum(decisionVariableY[i,j,k] for j in range(row) for k in range(row)) == 0 
      for j in range(row):
        problem += (decisionVariableY[i,i,j] == 0) # elimination of (1 to 1 to 2) route for drone
        problem += (decisionVariableY[i,j,i] == 0) # elimination of (1 to 2 to 1) route for drone
        problem += (decisionVariableY[j,i,i] == 0) # elimination of (1 to 2 to 2) route for drone
        if i != j and (i != 0 and j != 0):
          problem += u[i] - u[j] <= M * (1 - decisionVariableX[i, j])-1 # sub-tour elimination for truck
        if i != j and (j != 0):
          problem+= decisionVariableT[j] >= decisionVariableT[i] + road_matrix.iloc[i,j] - M*(1-decisionVariableX[i,j])

    #  each node j is visit once by truck or drone  
    for j in range(row):
        problem += lpSum(decisionVariableX[i,j] for i in range(row)) + lpSum(decisionVariableY[i,j,k] for i in range(row) for k in range(row)) == 1

    # if truck arrives on node i it leave node i (flow conservation constraint)
    for i in range(row):
      problem += lpSum(decisionVariableX[i,j] for j in range(row)) == lpSum(decisionVariableX[j,i] for j in range(row))

    # drone is dispatched from a node once, it is received at a node once
    for i in range(row):
      problem += lpSum(decisionVariableY[i,j,k] for j in range(row) for k in range(row)) <=1
      problem += lpSum(decisionVariableY[k,j,i] for k in range(row) for j in range(row)) <=1
    

    for i in range(row):
      for j in range(row):
        problem += decisionVariableD[i] >= decisionVariableT[i] - M*(1-lpSum(decisionVariableY[i,k,j] for k in range(row)))
        problem += decisionVariableD[j] >= decisionVariableT[j] - M*(1-lpSum(decisionVariableY[i,k,j] for k in range(row)))
        problem += decisionVariableD[i] <= decisionVariableT[i] + M*(1-lpSum(decisionVariableY[i,k,j] for k in range(row)))
        problem += decisionVariableD[j] <= decisionVariableT[j] + M*(1-lpSum(decisionVariableY[i,k,j] for k in range(row)))

    # if drone is dispatched or received from a node, the truck is also assigned to that node
    for i in range(row):
      for j in range(row):
        for k in range(row):
          if (k==0):
            problem += decisionVariableY[i,j,k] == 0
          problem += lpSum(decisionVariableX[i,a] for a in range(row)) >= decisionVariableY[i,j,k] 
          problem += lpSum(decisionVariableX[k,a] for a in range(row)) >= decisionVariableY[i,j,k] 


    # Calculating drone travel time
    for i in range(row):
      for j in range(row):
        for k in range(row):
          if (i!=j) and (j!=k):
            problem += decisionVariableD[j] >= decisionVariableD[i] + drone_matrix.iloc[i,j] - M*(1-decisionVariableY[i,j,k])
            problem += decisionVariableD[j] >= decisionVariableD[i] + drone_matrix.iloc[i,j] - M*(1-decisionVariableY[k,i,j])
            problem += decisionVariableD[j] <= decisionVariableD[i] + drone_matrix.iloc[i,j] + M*(1-decisionVariableY[i,j,k])
            problem += decisionVariableD[j] <= decisionVariableD[i] + drone_matrix.iloc[i,j] + M*(1-decisionVariableY[k,i,j])

    # avoid sub tour for drones
    for i in range(row):
      for j in range(row):
        if (i!=j) and (i!=0):
          problem += v[i] - v[j] <= M * (1 - lpSum(decisionVariableY[i,k,j] for k in range(row)))-1

    # pij check
    for i in range(row):
      for j in range(row):
        if (i!=j):
          problem += p[i,j] + p[j,i] == 1
        if (i == 0):
          problem += p[i,j] == 1

    # pij check 2
    for i in range(row):
      for k in range(row):
        for l in range(row):
          if (i != k) and (i != l) and (k != l):
            problem += decisionVariableD[l] >= decisionVariableD[k] - M*(3 - lpSum(decisionVariableY[i,j,k] for j in range(row)) - lpSum(decisionVariableY[l,m,n] for m in range(row) for n in range(row))  - p[i,l])

    # last stop time
    for i in range(row):
      problem += decisionVariableT[i] <= z

    status = problem.solve(CPLEX_CMD(msg = 0)) 
    for var in problem.variables():
        if (var.value() >1):
            result.append(var.value())
            result_name.append(var.name)
    result_df['Variable Name'] = result_name
    result_df['Variable Value'] = result
    result_df['CPU time'] = problem.solutionTime
    result_df['Objective Value'] = problem.objective.value()

    # creating route list      
    count = 0
    drone_route=[]
    for i in range(row):
      for j in range(row):
          if (pulp.value(decisionVariableX[i,j])==1):
              count = count+1
          for k in range(row):
              if (pulp.value(decisionVariableY[i,j,k])==1):
                  drone_route.append(i)
                  drone_route.append(j)
                  drone_route.append(k)
    route = [0]*count
    for x,value in enumerate(route):
      for j in range(row):
          if (pulp.value(decisionVariableX[value,j])==1):
              if (j!=0):
                route[x+1] = j
    route.append(0)

    last_stop=pulp.value(decisionVariableT[route[-2]]) + road_matrix.iloc[route[-2],0]

    return(route, drone_route, problem.objective.value())
