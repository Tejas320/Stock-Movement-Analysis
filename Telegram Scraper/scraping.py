from telethon.sync import TelegramClient
import pandas as pd

api_id = 'XXXXXXXX'  
api_hash = 'XXXXXXXXXXXXXXXX'  
channel_username = 'StockPhoenix'

# Create a client
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    # Connect to the client
    async with client:
        # Get the channel entity
        channel = await client.get_entity(channel_username)

        # Initialize a list to store messages
        messages_data = []

        # Fetch 1000 messages from the channel
        async for message in client.iter_messages(channel, limit=50000):
            messages_data.append({
                'Date': message.date,
                'Message': message.text,
                'Message_ID': message.id,
                'Views': message.views,
                'Forwards': message.forwards,
                'Reactions': message.reactions
            })

        # Create a DataFrame from the collected data
        df = pd.DataFrame(messages_data)

        # Save the DataFrame to a CSV file
        df.to_csv('telegram_channel_messages.csv', index=False)

        print("Messages have been saved to 'telegram_channel_messages.csv'.")

# Run the main function
with client:
    client.loop.run_until_complete(main())
