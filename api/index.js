const Busboy = require('busboy');
const tf = require('@tensorflow/tfjs');
const nsfw = require('nsfwjs');
const jpeg = require('jpeg-js');
const png = require('pngjs').PNG;

// نمنع Vercel من معالجة الجسم تلقائياً لنستطيع قراءة الصورة كـ Stream
export const config = {
  api: {
    bodyParser: false,
  },
};

// متغير لتخزين الموديل في الذاكرة (Caching) حتى لا نعيد تحميله مع كل طلب
let _model;

const loadModel = async () => {
  if (_model) {
    return _model;
  }
  // تحميل الموديل بحجم صغير مناسب للسيرفرات
  _model = await nsfw.load();
  return _model;
};

// دالة لتحويل الصورة (Buffer) إلى Tensor
const imageToTensor = (buffer, type) => {
  let pixels;
  let width, height;

  if (type === 'image/png') {
    const pngImage = png.sync.read(buffer);
    width = pngImage.width;
    height = pngImage.height;
    pixels = pngImage.data;
  } else {
    // نفترض أنها JPEG كخيار افتراضي
    const jpegImage = jpeg.decode(buffer, { useTArray: true });
    width = jpegImage.width;
    height = jpegImage.height;
    pixels = jpegImage.data;
  }

  // إنشاء Tensor من مصفوفة البيكسلات
  // الرقم 3 يعني (RGB)، نحتاج لحذف قناة الشفافية (Alpha) إذا وجدت لأن الموديل يتدرب على 3 قنوات
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
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
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
        res.status(400).json({ error: 'No image file uploaded.' });
        return resolve();
      }

      try {
        const model = await loadModel();
        
        // تحويل الصورة وتحليلها
        const tensor = imageToTensor(fileBuffer, mimeType);
        const predictions = await model.classify(tensor);
        
        // تنظيف الذاكرة
        tensor.dispose();

        res.status(200).json(predictions);
        resolve();
      } catch (error) {
        console.error('Error processing image:', error);
        res.status(500).json({ error: 'Internal Server Error', details: error.message });
        resolve();
      }
    });

    req.pipe(busboy);
  });
}
