echo "Executing stuff"

time python update_data.py

echo "Finished updating"

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"ryanpdwyer@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml