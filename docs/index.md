# Snake Math

Interactive mathematical concepts powered by Python in your browser.

<script type="text/javascript" src="https://pyscript.net/releases/2024.1.1/core.js"></script>

## Try It Now

<py-config>
packages = ["numpy", "matplotlib"]
</py-config>

<div>
  <label>Enter a value: </label>
  <input type="range" id="slider" min="0" max="10" value="5" />
  <span id="output">5</span>
</div>

<py-script>
from pyscript import document
import numpy as np

def update_output(event=None):
    value = document.querySelector("#slider").value
    result = int(value) ** 2
    document.querySelector("#output").innerText = f"{value}² = {result}"

# Bind event
document.querySelector("#slider").addEventListener("input", update_output)
update_output()
</py-script>
