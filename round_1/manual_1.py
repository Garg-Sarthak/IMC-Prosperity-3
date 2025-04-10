
mx = 0
Gpath = []
N = 4

def strategy (grid,currCom,curr,n,path:list):
    global mx
    global Gpath
    if (n == 0) :
        if (currCom != 3):
            return
        
        if (curr > mx):
            mx = curr
            Gpath = path.copy()
        return
    for i in range (0,N):
        # if i == currCom-1 :
        #     continue
        path.append(i)
        strategy(grid,i,curr*grid[currCom][i],n-1,path)
        path.pop()


grid = [[1,1.45,0.52,0.72],[0.7,1,0.31,0.48],[1.95,3.1,1,1.49],[1.34,1.98,0.64,1]]
strategy(grid,3,1,5,[])
print(mx,Gpath)
print (2_000_000*mx)
