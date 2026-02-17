import numpy as np

# =============================================================================
# 1. BEWERTUNGS-STRATEGIEN
# =============================================================================
class Wertung:
    """Basisklasse für alle Bewertungsstrategien."""
    def calculate_score(self, board, player):
        raise NotImplementedError

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
# 2. DIE SPIEL-UMGEBUNG
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
            # === GEÄNDERTE LOGIK MIT UNENTSCHIEDEN ===
            stuck_player = self.current_player
            stuck_player_score = self.wertung_strategie.calculate_score(self.board, stuck_player)
            
            winner = 0 # 0 für Unentschieden
            if stuck_player_score > 15:
                winner = stuck_player
            elif stuck_player_score < 15:
                winner = stuck_player * -1

            # Belohnung für den Spieler, der den letzten Zug gemacht hat (previous_player)
            if winner == 0:
                reward = 0
            elif winner == previous_player:
                reward = 1
            else:
                reward = -1
            
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

if __name__ == '__main__':
    env = TokanEnvironment(wertung_strategie=HoehenWertung())
    print("--- Startzustand ---")
    env.render()
    done = False
    turn_count = 1
    max_turns = 200
    while not done and turn_count <= max_turns:
        print(f"\n<<< ZUG {turn_count} >>>")
        legal_moves = env.get_all_legal_moves(env.current_player)
        if not legal_moves:
            print(f"Spieler {env.current_player} kann keinen Zug machen. Spiel vorbei!")
            done = True
            continue
        action = legal_moves[np.random.randint(0, len(legal_moves))]
        print(f"Spieler {'Plus (+)' if env.current_player == 1 else 'Minus (-)'} wählt Aktion: {action}")
        board_state, reward, done = env.step(action)
        print("--- Zustand nach dem Zug ---")
        env.render()
        turn_count += 1
    
    print("\n" + "="*30)
    print("====== S P I E L E N D E ======")
    print("="*30)
    
    # === GEÄNDERTE LOGIK MIT UNENTSCHIEDEN ===
    stuck_player = env.current_player
    score = env.wertung_strategie.calculate_score(env.board, stuck_player)
    print(f"\nSpieler {stuck_player} kann nicht ziehen. Sein finaler Score ist: {score}")
    
    winner = 0
    if score > 15:
        winner = stuck_player
    elif score < 15:
        winner = stuck_player * -1
    
    if winner == 0:
        print("\nDas Spiel endet unentschieden!")
    else:
        winner_str = "'Plus (+)'" if winner == 1 else "'Minus (-)'"
        print(f"\nDer Gewinner ist Spieler {winner_str}!")