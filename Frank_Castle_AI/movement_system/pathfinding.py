import heapq

def heuristic(a, b):
    # Manhattan distance as heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    """
    grid: 2D list representing the map (0 = free, 1 = obstacle)
    start, goal: (x, y) tuples
    """
    neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start: heuristic(start, goal)}
    oheap = [(fscore[start], start)]
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]  # reverse path

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1  # assume cost=1 for each step

            if (0 <= neighbor[0] < len(grid)) and (0 <= neighbor[1] < len(grid[0])):
                if grid[neighbor[0]][neighbor[1]] == 1:
                    # 1 represents an obstacle
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []  # no path found

def plan_in_game_action(player_state):
    """
    Decide on in-game actions such as buying doors, perks, reloading, or shooting.
    `player_state` can include available funds, current ammo, perks status, etc.
    This function is a placeholder. Expand it with game-specific logic.
    """
    # Example decision-making logic (stub)
    if player_state.get('ammo', 0) < 5:
        return "reload"
    elif player_state.get('funds', 0) >= 100:
        return "buy_door"
    else:
        return "advance"

# For testing purposes
if __name__ == "__main__":
    # Example grid (0 free, 1 obstacle)
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    start = (0, 0)
    goal = (2, 3)
    print("Path:", astar(grid, start, goal))
    # Example player state
    player_state = {'ammo': 3, 'funds': 120}
    print("Action:", plan_in_game_action(player_state))
