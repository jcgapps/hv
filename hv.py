import logging
import math
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configura el logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# Diccionario para almacenar datos temporales de los usuarios
user_data = {}

# Comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("¡Hola! Envíame tu capital inicial:")

# Manejar mensajes de texto
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    text = update.message.text.strip()

    if chat_id not in user_data:
        # Guardar el capital inicial y pedir el capital final
        try:
            user_data[chat_id] = {"capital_inicial": float(text)}
            await update.message.reply_text("Ahora envíame tu capital objetivo:")
        except ValueError:
            await update.message.reply_text("Por favor, envía un número válido para el capital inicial.")
    elif "capital_final" not in user_data[chat_id]:
        # Guardar el capital final y calcular los días
        try:
            user_data[chat_id]["capital_final"] = float(text)
            dias = calcular_dias(user_data[chat_id]["capital_inicial"], user_data[chat_id]["capital_final"])
            await update.message.reply_text(f"Tardarás aproximadamente {round(dias, 2)} días en alcanzar {user_data[chat_id]['capital_final']}.")
            del user_data[chat_id]  # Limpiar datos del usuario
        except ValueError:
            await update.message.reply_text("Por favor, envía un número válido para el capital objetivo.")

# Función para calcular los días
def calcular_dias(capital_inicial, capital_final):
    r = 0.0172  # Crecimiento diario del 1.72%
    return math.log(capital_final / capital_inicial) / math.log(1 + r)

# Función principal para ejecutar el bot
def main():
    TOKEN = "7668316935:AAHTPJqj5waAhMlbDa99bKx8N2moBvZ-TUw"  # Token del bot (¡CAMBIA ESTO INMEDIATAMENTE!)
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot en marcha...")
    app.run_polling()

if __name__ == "__main__":
    main()
