# stock_trend_analysis

AI-Powered Stock Trend Analysis Tool
A sophisticated stock analysis tool built with Python that combines technical analysis, sentiment analysis, and machine learning to provide comprehensive stock trading insights.
Features

Real-time Stock Data Analysis: Fetches and analyzes current stock data using yfinance
Technical Indicators:

Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
Buy/Sell signal generation


Sentiment Analysis:

News sentiment analysis using TextBlob
Social media sentiment tracking (Twitter, Reddit)
Combined sentiment scoring


Interactive Visualization:

Candlestick charts with Plotly
Buy/Sell signals overlay
Technical indicator graphs


Trading Strategies:

Intraday trading recommendations
Swing trading insights
Customizable timeframes



Installation

Clone the repository:

bashCopygit clone https://github.com/uday8073262808/stock_trend_analysis.git
cd stock_trend_analysis

Create and activate a virtual environment:

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install required packages:

bashCopypip install -r requirements.txt
Dependencies

streamlit
yfinance
pandas
plotly
numpy
textblob
requests

Usage

Start the Streamlit application:

bashCopystreamlit run app.py

Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
Select your desired timeframe
Choose your trading strategy preference
View the analysis results including:

Interactive price charts
Technical indicators
Sentiment analysis
Trading recommendations



Features Detail
Technical Analysis

RSI (Relative Strength Index): Measures overbought/oversold conditions
MACD (Moving Average Convergence Divergence): Identifies trend changes and momentum
Buy/Sell Signals: Generated based on combined technical indicators

Sentiment Analysis

News sentiment scoring
Social media sentiment tracking
Combined sentiment analysis with weighted scoring

Interactive Visualization

Candlestick charts for price movement
Technical indicator overlays
Buy/Sell signal markers
Downloadable technical analysis data
