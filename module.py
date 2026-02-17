import numpy as np

# --- Aktivierungsfunktionen und ihre Ableitungen ---
def sigmoid(x):
    """Aktivierungsfunktion: quetscht Werte in den Bereich zwischen 0 und 1."""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """Ableitung der Sigmoid-Funktion."""
    s = sigmoid(x)
    return s * (1 - s)

# --- Loss-Funktion und ihre Ableitung ---
def mse(y_true, y_pred):
    """Mean Squared Error: misst den durchschnittlichen quadratischen Fehler."""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """Ableitung der Mean Squared Error-Funktion."""
    return 2 * (y_pred - y_true) / np.size(y_true)

# --- Basis-Klasse für alle Layer ---
class Layer:
    """Grundgerüst für alle Schichten im Netzwerk."""
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """Berechnet den Output der Schicht (Forward Pass)."""
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        """Berechnet den Gradienten und passt Gewichte an (Backward Pass)."""
        raise NotImplementedError

# --- Schicht für lineare Operationen (Gewichte und Bias) ---
class Dense(Layer):
    """Eine voll verbundene Schicht (y = Wx + b)."""
    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialisiert Gewichte mit kleinen zufälligen Werten
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.random.randn(output_size, 1) * 0.1

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Berechnet Gradienten für Gewichte und Bias
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = output_gradient # Vereinfachung für einzelne Samples
        
        # Berechnet den Gradienten für die vorherige Schicht
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Aktualisiert die Gewichte und den Bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient

# --- Schicht für Aktivierungsfunktionen ---
class Activation(Layer):
    """Eine Schicht, die eine Aktivierungsfunktion anwendet."""
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Berechnet den Gradienten durch die Aktivierungsfunktion
        return output_gradient * self.activation_prime(self.input)

# --- Hauptklasse für das Neuronale Netzwerk ---
class Network:
    """Die Klasse, die das Netzwerk verwaltet und den Trainingsprozess steuert."""
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        """Fügt eine Schicht zum Netzwerk hinzu."""
        self.layers.append(layer)
        
    def predict(self, input_data):
        """Macht eine Vorhersage für gegebene Input-Daten."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate):
        """Trainiert das Netzwerk."""
        for i in range(epochs):
            total_error = 0
            for x, y in zip(x_train, y_train):
                # 1. Forward Pass
                output = self.predict(x)
                
                # 2. Fehler berechnen
                total_error += mse(y, output)
                
                # 3. Backward Pass
                gradient = mse_prime(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
            
            # Gibt den durchschnittlichen Fehler für die Epoche aus
            avg_error = total_error / len(x_train)
            if (i + 1) % 100 == 0: # Nur alle 100 Epochen ausgeben
                print(f"Epoche {i + 1}/{epochs}, Fehler: {avg_error:.5f}")

# --- Hauptprogramm: Netzwerk erstellen und trainieren ---
if __name__ == '__main__':
    # Trainingsdaten für das XOR-Problem
    x_train = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # Netzwerk-Architektur definieren
    net = Network()
    net.add(Dense(2, 3))  # Input-Schicht: 2 Neuronen -> Hidden-Schicht: 3 Neuronen
    net.add(Activation(sigmoid, sigmoid_prime))
    net.add(Dense(3, 1))  # Hidden-Schicht: 3 Neuronen -> Output-Schicht: 1 Neuron
    net.add(Activation(sigmoid, sigmoid_prime))

    # Netzwerk trainieren
    print("--- Starte Training ---")
    net.train(x_train, y_train, epochs=1000, learning_rate=0.1)
    print("--- Training beendet ---")

    # Ergebnisse nach dem Training testen
    print("\n--- Testergebnisse ---")
    for x, y in zip(x_train, y_train):
        prediction = net.predict(x)
        print(f"Input: {x.flatten()}, Erwartet: {y.flatten()[0]}, Vorhergesagt: {prediction[0][0]:.4f}")