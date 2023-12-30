import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuCNN(nn.Module):
    def __init__(self, dropout_prob=0.3):  
        super(GomokuCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 15 * 15, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop_fc = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)  
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x.squeeze()
    


class Submission:

    def __init__(self, board_size, win_size):
        self.board_size = board_size
        self.win_size = win_size
        self.model = GomokuCNN()
        self.model.load_state_dict(torch.load('gomoku_cnn_model_final.pth'))
        self.model.eval()
        

    def get_heuristic_from_model(self, board):
        """Get heuristic score for a given board state from the model."""
        prepared_board = self.prepare_board([board])
        with torch.no_grad():
            predicted_score = self.model(prepared_board)
        return predicted_score.item()
    
    def get_monotonic_array(self, state):
        size=self.board_size  
        monotonic_array = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                if state.board[0, i, j] == 1:
                    monotonic_array[i, j] = 0
                elif state.board[1, i, j] == 1:
                    monotonic_array[i, j] = 1
                elif state.board[2, i, j] == 1:
                    monotonic_array[i, j] = 2

        return monotonic_array

    def is_valid_position(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def get_sequence(self, r, c, dr, dc, board, length=4):
        return [board[r + i * dr][c + i * dc] for i in range(length) if self.is_valid_position(r + i * dr, c + i * dc)]

    def prepare_board(self, board):
        tensor_board = torch.tensor(board).float()
        tensor_board = tensor_board.unsqueeze(0) / 2  
        return tensor_board

    def report_threat(self, threat_type, position, action):
        print(f"Threat detected: {threat_type} at position {position}. Taking action: {action}")

    def find_threats_and_counter_actions(self, board, patterns):
        threats = []
        counter_actions = []

        for i in range(self.board_size):
            for j in range(self.board_size):
                # Check for patterns of length 4 and 5
                for length in [5, 4]:
                    # Check horizontally, vertically, diagonal down-right, and diagonal down-left
                    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if self.is_valid_position(i + (length - 1) * dr, j + (length - 1) * dc):
                            seq = self.get_sequence(i, j, dr, dc, board, length)

                            for pattern in patterns:
                                if seq == pattern:
                                    zero_index = pattern.index(0)
                                    threat_position = (i + zero_index * dr, j + zero_index * dc)
                                    threats.append(seq)  
                                    counter_actions.append(threat_position)
        return threats, counter_actions
    
    def find_win_moves(self, board, win_pattern):
        win_actions = []

        for i in range(self.board_size):
            for j in range(self.board_size):
                # Check horizontally, vertically, diagonal down-right, and diagonal down-left
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if self.is_valid_position(i + 4 * dr, j + 4 * dc):
                        seq = self.get_sequence(i, j, dr, dc, board, 5)
                        if seq == win_pattern:
                            win_index = win_pattern.index(0)
                            win_position = (i + win_index * dr, j + win_index * dc)
                            win_actions.append(win_position)
        return win_actions
    
    def __call__(self, state):
        monotonic_array = self.get_monotonic_array(state)
        actions = state.valid_actions()
        board = monotonic_array

        # Can we win in this move?
        win_pattern = [1, 1, 1, 1, 0]
        win_actions = self.find_win_moves(monotonic_array, win_pattern)

        # Is the opponent winning the next move?
        opp_pattern = [2, 2, 2, 2, 0]
        opp_actions = self.find_win_moves(monotonic_array, opp_pattern)

        # If no immidiate threats then attack by building open 3
        if len(opp_actions)==0:
            win_pattern_2=[0,1,1,1,0]
            win_actions_2=self.find_win_moves(monotonic_array, win_pattern)
            win_actions=win_actions+win_actions_2
        
        if win_actions:
            #print(f"Winning move found at {win_actions[0]}")
            children=[]
            for (r,c) in win_actions:
                a=monotonic_array.copy()
                a[r,c]=1
                children.append(a)
            heuristics=[self.get_heuristic_from_model(child) for child in children] 
            minH=np.argmin(heuristics)
            #return win_actions[0] #Choose the action that was first added to the list
            return win_actions[minH] #Or Play by heuristics
        
        patterns = [
            [0, 2, 2, 2], [2, 0, 2, 2], [2, 2, 0, 2], [2, 2, 2, 0], 
             [0,2,2,2,2],[2, 2, 2, 0, 2],[2,0,2,2,2],[2,2,0,2,2], [2,2,2,2,0] ,[2,0,2,0,2],[0,2,2,2,0] 
        ]

        threats, counter_actions = self.find_threats_and_counter_actions(board, patterns)
        if threats!=[]:
            zipped_list = zip(threats, counter_actions)
            sorted_list = sorted(zipped_list, key=lambda x: sum(float(num) for num in x[0]), reverse=True)
            threats_sorted, counter_actions_sorted = zip(*sorted_list)
            threats = list(threats_sorted)
            counter_actions = list(counter_actions_sorted)

        
        # Handle threats
        # If there is only 1 threat, return that action the corrective action.
        # If there are multiple threats then check if there is a serious threat aka opponent is winning in the next move.
        # If no serious threat then use the model to get a heuristic for each move and then pick the move with the minimum heuristic value. 
            
            if len(threats) == 1:
                #print(f"Single threat detected: {threats[0]}. Taking counter-action at {counter_actions[0]}")
                return counter_actions[0]
            elif len(threats) > 1:
                if all(action == counter_actions[0] for action in counter_actions):
                    #print(f"Multiple threats detected, but a single counter-action is effective at {counter_actions[0]}")
                    return counter_actions[0]
                elif len(threats[0])==5:
                    #print(f"Serious Threat Detected: {threats[0]}. Taking counter-action at {counter_actions[0]}")
                    return counter_actions[0]
                else:
                    #print(f"Multiple threats detected. Unable to respond effectively.")
                    #for threat, action in zip(threats, counter_actions):
                        #print(f"Detected threat: {threat}, Counter-action: {action}")
                    children=[]
                    for (r,c) in counter_actions:
                        a=monotonic_array.copy()
                        a[r,c]=1
                        children.append(a)
                    heuristics=[self.get_heuristic_from_model(child) for child in children] 
                    minH=np.argmin(heuristics)
                    #print(f"Serious Threat Detected: {threats[minH]}. Taking counter-action at {counter_actions[minH]}")         
                    return counter_actions[minH] #Consider Heuristics
                    #return counter_actions[0] #No heuristics
                
        # If no winning moves, no threats, no open 3s, evaluate heuristic for each move and select the move with minimum heuristic.
        else:        
            children=[]
            for (r,c) in actions:
                a=monotonic_array.copy()
                a[r,c]=1
                children.append(a)
            heuristics=[self.get_heuristic_from_model(child) for child in children]
            
            minH=np.argmin(heuristics)
            #print("No threat. Heuristic move",actions[minH])  
            # return actions[-1]    #No heuristics  
            return actions[minH]
