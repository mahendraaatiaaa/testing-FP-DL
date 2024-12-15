const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

// Menyajikan folder static
app.use(express.static(path.join(__dirname, 'static')));

// Menyajikan folder models
app.use('/models', express.static(path.join(__dirname, 'models')));

// Halaman utama
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
