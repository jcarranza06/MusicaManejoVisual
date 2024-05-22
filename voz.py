import pyttsx3

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

# Configurar propiedades del habla
engine.setProperty('rate', 150)  # Velocidad de habla (palabras por minuto)
engine.setProperty('volume', 1)  # Volumen (0.0 a 1.0)

# Función para hablar un texto
def hablar(texto):
    engine.say(texto)
    engine.runAndWait()

# Ejemplo de uso
hablar("Hola, bienvenido al programa de Python con voz.")
hablar("Este es un ejemplo de cómo agregar sonido a tu programa.")
