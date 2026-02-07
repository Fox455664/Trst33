const Busboy = require('busboy');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu'); // 👈 هذا هو السطر السحري للإصلاح
const nsfw = require('nsfwjs');
const jpeg = require('jpeg-js');
const png = require('pngjs').PNG;

// ضبط إعدادات Vercel
export const config = {
  api: {
    bodyParser: false, // تعطيل الـ Body Parser لاستقبال الملفات
  },
};

// متغير لتخزين الموديل
let _model;

const loadModel = async () => {
  if (_model) {
    return _model;
  }
  // تعريف المعالج صراحةً لتجنب المشاكل
  await tf.setBackend('cpu');
  console.log('Loading model...');
  _model = await nsfw.load(); 
  return _model;
};

const imageToTensor = (buffer, type) => {
  let pixels;
  let width, height;

  if (type === 'image/png') {
    const pngImage = png.sync.read(buffer);
    width = pngImage.width;
    height = pngImage.height;
    pixels = pngImage.data;
  } else {
    const jpegImage = jpeg.decode(buffer, { useTArray: true });
    width = jpegImage.width;
    height = jpegImage.height;
    pixels = jpegImage.data;
  }

  const numChannels = 3;
  const numPixels = width * height;
  const values = new Int32Array(numPixels * numChannels);

  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel];
    }
  }

  return tf.tensor3d(values, [height, width, numChannels], 'int32');
};

export default async function handler(req, res) {
  // 1. التعامل مع زيارة المتصفح العادية (GET)
  if (req.method === 'GET') {
    return res.status(200).send('<h1>Server is Running ✅</h1><p>Please send a POST request with an image to verify.</p>');
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const busboy = Busboy({ headers: req.headers });
  let fileBuffer = null;
  let mimeType = '';

  return new Promise((resolve) => {
    busboy.on('file', (fieldname, file, info) => {
      const { mimeType: type } = info;
      mimeType = type;
      const chunks = [];
      file.on('data', (data) => chunks.push(data));
      file.on('end', () => {
        fileBuffer = Buffer.concat(chunks);
      });
    });

    busboy.on('finish', async () => {
      if (!fileBuffer) {
        res.status(400).json({ error: 'No image uploaded' });
        return resolve();
      }

      try {
        const model = await loadModel();
        const tensor = imageToTensor(fileBuffer, mimeType);
        const predictions = await model.classify(tensor);
        tensor.dispose();

        res.status(200).json(predictions);
        resolve();
      } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: error.message });
        resolve();
      }
    });

    req.pipe(busboy);
  });
}
