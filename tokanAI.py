import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import argparse
from collections import deque
from copy import deepcopy

# =============================================================================
# 1. BEWERTUNGS-STRATEGIEN
# =============================================================================
class Wertung:
    """Basisklasse für alle Bewertungsstrategien."""
    def calculate_score(self, board, player):
        raise NotImplementedError("Diese Methode muss in der Subklasse implementiert werden.")

class HoehenWertung(Wertung):
    """Bewertet den Punktestand basierend auf der Höhe der kontrollierten Stapel."""
    def _get_stack_height(self, stack):
        non_zero_indices = np.where(stack != 0)[0]
        return len(non_zero_indices)

    def calculate_score(self, board, player):
        total_score = 0
        max_y, max_x, _ = board.shape
        for y in range(max_y):
            for x in range(max_x):
                stack = board[y, x]
                stack_height = self._get_stack_height(stack)
                if stack_height > 0:
                    top_piece = stack[stack_height - 1]
                    if np.sign(top_piece) == player:
                        total_score += stack_height
        return total_score

# =============================================================================
# 2. DIE SPIEL-UMGEBUNG (TOKIAN ENVIRONMENT)
# =============================================================================
class TokanEnvironment:
    def __init__(self, wertung_strategie=HoehenWertung()):
        self.max_y, self.max_x, self.max_stack_height = 6, 5, 30
        self.board = np.zeros((self.max_y, self.max_x, self.max_stack_height), dtype=int)
        self.pieces = [1]*3 + [-1]*3 + [2]*7 + [-2]*7 + [3]*5 + [-3]*5
        self.current_player = 1
        self.wertung_strategie = wertung_strategie
        self.last_move = None
        self.reset()

    def reset(self):
        pieces_to_place = self.pieces[:]
        np.random.shuffle(pieces_to_place)
        self.board = np.zeros((self.max_y, self.max_x, self.max_stack_height), dtype=int)
        for i in range(len(pieces_to_place)):
            y, x = i // self.max_x, i % self.max_x
            self.board[y, x, 0] = pieces_to_place[i]
        self.current_player = np.random.choice([1, -1])
        self.last_move = None
        return self.board

    def get_top_piece_info(self, y, x):
        stack = self.board[y, x]
        non_zero_indices = np.where(stack != 0)[0]
        if len(non_zero_indices) == 0: return 0, 0
        height = len(non_zero_indices)
        top_piece_value = stack[height - 1]
        return top_piece_value, height

    def get_all_legal_moves(self, player):
        legal_moves = []
        for y_from in range(self.max_y):
            for x_from in range(self.max_x):
                top_piece, h_from = self.get_top_piece_info(y_from, x_from)
                if np.sign(top_piece) != player: continue
                distance = abs(top_piece)
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for dy, dx in directions:
                    y_to, x_to = y_from + dy * distance, x_from + dx * distance
                    from_pos, to_pos = (y_from, x_from), (y_to, x_to)
                    max_takeable = abs(top_piece)
                    for num_pieces in range(1, max_takeable + 1):
                        if self.is_move_valid(from_pos, to_pos, h_from, num_pieces):
                            legal_moves.append((from_pos, to_pos, num_pieces))
        return legal_moves
        
    def is_move_valid(self, from_pos, to_pos, h_from, num_pieces):
        y_to, x_to = to_pos
        if h_from < num_pieces: return False
        if not (0 <= y_to < self.max_y and 0 <= x_to < self.max_x): return False
        y_from, x_from = from_pos
        path_len = max(abs(y_to - y_from), abs(x_to - x_from))
        dy, dx = np.sign(y_to - y_from), np.sign(x_to - x_from)
        for i in range(1, path_len):
            y_step, x_step = y_from + i * dy, x_from + i * dx
            if self.get_top_piece_info(y_step, x_step)[1] == 0: return False
        _, h_to = self.get_top_piece_info(y_to, x_to)
        if h_to == 0 or not (h_to + num_pieces > h_from): return False
        return True
    
    def _close_gap(self, y_from, x_from, y_to, x_to):
        dy, dx = np.sign(y_to - y_from), np.sign(x_to - x_from)
        if dy == -1:
            gap_y = y_from
            for current_y in range(y_from + 1, self.max_y):
                if self.get_top_piece_info(current_y, x_from)[1] > 0:
                    self.board[gap_y, x_from, :] = self.board[current_y, x_from, :].copy()
                    self.board[current_y, x_from, :] = 0
                    gap_y += 1
        elif dy == 1:
            gap_y = y_from
            for current_y in range(y_from - 1, -1, -1):
                if self.get_top_piece_info(current_y, x_from)[1] > 0:
                    self.board[gap_y, x_from, :] = self.board[current_y, x_from, :].copy()
                    self.board[current_y, x_from, :] = 0
                    gap_y -= 1
        elif dx == -1:
            gap_x = x_from
            for current_x in range(x_from + 1, self.max_x):
                if self.get_top_piece_info(y_from, current_x)[1] > 0:
                    self.board[y_from, gap_x, :] = self.board[y_from, current_x, :].copy()
                    self.board[y_from, current_x, :] = 0
                    gap_x += 1
        elif dx == 1:
            gap_x = x_from
            for current_x in range(x_from - 1, -1, -1):
                if self.get_top_piece_info(y_from, current_x)[1] > 0:
                    self.board[y_from, gap_x, :] = self.board[y_from, current_x, :].copy()
                    self.board[y_from, current_x, :] = 0
                    gap_x -= 1

    def step(self, action):
        from_pos, to_pos, num_pieces = action
        y_from, x_from = from_pos
        y_to, x_to = to_pos
        
        _, h_from = self.get_top_piece_info(y_from, x_from)
        start_idx, end_idx = h_from - num_pieces, h_from
        stack_to_move = self.board[y_from, x_from, start_idx:end_idx].copy()
        
        self.board[y_from, x_from, start_idx:end_idx] = 0
        
        _, h_to = self.get_top_piece_info(y_to, x_to)
        self.board[y_to, x_to, h_to:(h_to + num_pieces)] = stack_to_move
        
        if self.get_top_piece_info(y_from, x_from)[1] == 0:
            self._close_gap(y_from, x_from, y_to, x_to)
        
        self.last_move = action
        previous_player = self.current_player
        self.current_player *= -1
        
        next_legal_moves = self.get_all_legal_moves(self.current_player)
        done = len(next_legal_moves) == 0
        
        reward = 0
        if done:
            stuck_player = self.current_player
            stuck_player_score = self.wertung_strategie.calculate_score(self.board, stuck_player)
            winner = 0
            if stuck_player_score > 15: winner = stuck_player
            elif stuck_player_score < 15: winner = stuck_player * -1
            if winner == 0: reward = 0
            elif winner == previous_player: reward = 1
            else: reward = -1
            
        return self.board, reward, done

    def render(self):
        print("-" * 80)
        print(f"Spieler am Zug: {'Plus (+)' if self.current_player == 1 else 'Minus (-)'}")
        from_pos, to_pos = (None, None), (None, None)
        if self.last_move:
            from_pos, to_pos, _ = self.last_move
        for y in range(self.max_y):
            row_str = ""
            for x in range(self.max_x):
                stack = self.board[y, x]
                pieces_in_stack = stack[stack != 0]
                marker = " "
                if (y, x) == from_pos: marker = ">"
                elif (y, x) == to_pos: marker = "*"
                if len(pieces_in_stack) == 0:
                    cell_str = "[         ]"
                else:
                    pieces_str_list = [f"{'+' if p > 0 else ''}{p}" for p in pieces_in_stack]
                    cell_content = ",".join(pieces_str_list)
                    cell_str = f"[{cell_content}]"
                full_cell_str = f"{marker}{cell_str}"
                row_str += f"{full_cell_str:<16}"
            print(row_str)
        print("-" * 80)

# =============================================================================
# 3. DAS CONVOLUTIONAL NEURAL NETWORK (CNN)
# =============================================================================
class TokanNet(nn.Module):
    def __init__(self):
        super(TokanNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 5, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.clone().detach().float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(-1, 128 * 6 * 5) 
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# =============================================================================
# 4. REPLAY BUFFER, AGENT & HELPER FUNCTIONS
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, reward):
        self.buffer.append((state, reward))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

def train(args, device):
    model = TokanNet().to(device)
    if args.load_path:
        try:
            model.load_state_dict(torch.load(args.load_path, map_location=device))
            print(f"Modell von '{args.load_path}' geladen.")
        except Exception as e:
            print(f"Konnte Modell nicht laden: {e}. Starte neues Training.")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(10000)
    epsilon = 1.0
    model.train()
    
    for episode in range(args.episodes):
        env = TokanEnvironment()
        done = False
        episode_memory = []

        while not done:
            legal_moves = env.get_all_legal_moves(env.current_player)
            if not legal_moves:
                done = True
                continue
            
            action = random.choice(legal_moves) # Vereinfachung: Zufällige Züge im Training
            state_before = env.board.copy()
            _, reward, done = env.step(action)
            if done:
                # Belohnung gilt für den Spieler, der den Zug gemacht hat
                final_reward = reward if env.current_player == -1 else -reward
                episode_memory.append((state_before, final_reward))

        if episode_memory:
            final_reward = episode_memory[-1][1]
            for state, _ in episode_memory:
                replay_buffer.push(state, final_reward)

        if len(replay_buffer) > args.batch_size:
            batch = replay_buffer.sample(args.batch_size)
            states, rewards = zip(*batch)
            
            state_tensors = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            reward_tensors = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)

            predicted_values = model(state_tensors)
            loss = F.mse_loss(predicted_values, reward_tensors)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{args.episodes}, Replay Buffer: {len(replay_buffer)}")
            
    print("\nTraining abgeschlossen.")
    torch.save(model.state_dict(), args.save_path)
    print(f"Modell in '{args.save_path}' gespeichert.")

def get_ai_suggestions(model, env, device):
    model.eval()
    legal_moves = env.get_all_legal_moves(env.current_player)
    if not legal_moves: return []
    move_scores = []
    for move in legal_moves:
        temp_env = deepcopy(env)
        next_state, _, _ = temp_env.step(move)
        with torch.no_grad():
            value = model(next_state).item()
            move_scores.append((move, value))
    is_maximizing = env.current_player == 1
    move_scores.sort(key=lambda x: x[1], reverse=is_maximizing)
    return move_scores

def get_human_move(env):
    all_legal_moves = env.get_all_legal_moves(env.current_player)
    moves_by_start_pos = {}
    for move in all_legal_moves:
        from_pos = move[0]
        if from_pos not in moves_by_start_pos: moves_by_start_pos[from_pos] = []
        moves_by_start_pos[from_pos].append(move)

    start_pos = None
    while start_pos is None:
        try:
            pos_str = input("Gib die Start-Koordinate des gegnerischen Zugs ein (Format: y,x): ")
            y_from, x_from = map(int, pos_str.split(','))
            selected_pos = (y_from, x_from)
            if selected_pos in moves_by_start_pos:
                start_pos = selected_pos
            else:
                print("Fehler: Von diesem Feld kann kein gültiger Zug gemacht werden.")
        except (ValueError, IndexError):
            print("Fehler: Ungültiges Format. Bitte 'y,x' eingeben.")
    
    possible_moves = moves_by_start_pos[start_pos]
    print(f"\nMögliche Züge von {start_pos}:")
    for i, move in enumerate(possible_moves):
        to_pos, num_pieces = move[1], move[2]
        y, x = to_pos
        print(f"  {i+1}: Nach {y},{x} mit {num_pieces} Stein(en)")
    while True:
        try:
            choice_str = input("Wähle die Nummer des Ziels: ")
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(possible_moves):
                return possible_moves[choice_idx]
            else:
                print("Fehler: Ungültige Nummer.")
        except ValueError:
            print("Fehler: Bitte eine Zahl eingeben.")

def play(args, device):
    model = TokanNet().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Modell von '{args.model_path}' geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return
        
    env = TokanEnvironment()
    player_choice = input("Möchtest du als Spieler 'Plus' (1) oder 'Minus' (-1) spielen? ")
    human_player = 1 if player_choice == '1' else -1
    done = False
    
    while not done:
        env.render()
        legal_moves = env.get_all_legal_moves(env.current_player)
        if not legal_moves:
            print(f"Spieler {env.current_player} kann keinen Zug machen. Spiel vorbei!")
            done = True
            continue

        if env.current_player == human_player:
            print("Die KI analysiert die besten Züge für dich...")
            suggestions = get_ai_suggestions(model, env, device)
            print("\n--- Empfohlene Züge (Bester zuerst) ---")
            for i, (move, score) in enumerate(suggestions):
                print(f"{i+1}: Zug: {move}, Geschätzter Wert: {score:.4f}")
            best_move = suggestions[0][0]
            print(f"\nKI führt besten Zug aus: {best_move}")
            _, _, done = env.step(best_move)
        else:
            print(f"Der Gegner (Spieler {'Plus' if env.current_player == 1 else 'Minus'}) ist am Zug.")
            chosen_move = get_human_move(env)
            _, _, done = env.step(chosen_move)
            
    print("\n" + "="*30 + "\n====== S P I E L E N D E ======\n" + "="*30)
    stuck_player = env.current_player
    score = env.wertung_strategie.calculate_score(env.board, stuck_player)
    print(f"\nSpieler {stuck_player} kann nicht ziehen. Sein finaler Score ist: {score}")
    winner = 0
    if score > 15: winner = stuck_player
    elif score < 15: winner = stuck_player * -1
    if winner == 0: print("\nDas Spiel endet unentschieden!")
    else:
        winner_str = "'Plus (+)'" if winner == 1 else "'Minus (-)'"
        print(f"\nDer Gewinner ist Spieler {winner_str}!")

# =============================================================================
# 5. DAS SHELL-INTERFACE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Trainiere oder spiele Tokan mit einem neuronalen Netz.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_train = subparsers.add_parser(   'train', help="Trainiere das neuronale Netz.")
    parser_train.add_argument("--episodes", type=int, default=1000, help="Anzahl der Trainingspartien.")
    parser_train.add_argument("--lr", type=float, default=0.001, help="Lernrate.")
    parser_train.add_argument("--batch-size", type=int, default=64, help="Größe der Batches für das Training.")
    parser_train.add_argument("--load-path", type=str, default=None, help="Lade ein existierendes Modell.")
    parser_train.add_argument("--save-path", type=str, default="tokan_model.pth", help="Speichere das trainierte Modell hier.")
    
    parser_play = subparsers.add_parser('play', help="Spiele interaktiv gegen die KI.")
    parser_play.add_argument("--model-path", type=str, required=True, help="Pfad zum geladenen, trainierten Modell.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.command == 'train':
        train(args, device)
    elif args.command == 'play':
        play(args, device)

if __name__ == '__main__':
    main()