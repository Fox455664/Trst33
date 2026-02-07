// api/index.js
const nsfw = require('nsfwjs');
const tf = require('@tensorflow/tfjs-core');
require('@tensorflow/tfjs-backend-cpu'); // استخدام المعالج بدلاً من كرت الشاشة
const Jimp = require('jimp');
const multiparty = require('multiparty');

// متغير لتخزين الموديل في الذاكرة لتسريع الطلبات المتكررة
let _model;

const loadModel = async () => {
  if (_model) {
    return _model;
  }
  // تحميل الموديل من الروابط العامة الافتراضية لتقليل حجم الكود المرفوع
  _model = await nsfw.load();
  return _model;
};

const convertImage = async (imageBuffer) => {
  const image = await Jimp.read(imageBuffer);
  const width = image.bitmap.width;
  const height = image.bitmap.height;
  const pixelCount = width * height;
  
  const float32Data = new Float32Array(3 * pixelCount);
  let i = 0;
  
  image.scan(0, 0, width, height, (x, y, idx) => {
    float32Data[i++] = image.bitmap.data[idx + 0] / 255; // Red
    float32Data[i++] = image.bitmap.data[idx + 1] / 255; // Green
    float32Data[i++] = image.bitmap.data[idx + 2] / 255; // Blue
  });

  const tensor = tf.tensor3d(float32Data, [height, width, 3]);
  return tensor;
};

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST requests are allowed');
  }

  const form = new multiparty.Form();

  form.parse(req, async (err, fields, files) => {
    if (err) return res.status(500).json({ error: err.message });
    
    // تأكد من اسم الحقل الذي سترسل الصورة فيه، هنا افترضنا اسمه 'image'
    const file = files.image && files.image[0];

    if (!file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    try {
      const model = await loadModel();
      
      // قراءة الصورة وتحويلها
      const fs = require('fs');
      const imageBuffer = fs.readFileSync(file.path);
      const tensor = await convertImage(imageBuffer);

      const predictions = await model.classify(tensor);
      
      // تنظيف الذاكرة
      tensor.dispose();

      res.status(200).json(predictions);
      
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Failed to classify image' });
    }
  });
};
