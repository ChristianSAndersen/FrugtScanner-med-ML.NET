using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Text.Json;

namespace FruitApp.Services
{
    public class OnnxService
    {
        private InferenceSession _session;
        private Dictionary<string, Models.Product> _products;

        private string[] classNames = new[]
        {
            "apple",
            "banana",
            "grape",
            "kiwi",
            "lemon",
            "orange",
            "peach",
            "pineapple",
            "strawberry",
            "watermelon"
        };

        public async Task Init()
        {           
            using var modelStream = await FileSystem.OpenAppPackageFileAsync("best.onnx");
            using var ms = new MemoryStream();
            await modelStream.CopyToAsync(ms);
            _session = new InferenceSession(ms.ToArray());
           
            using var jsonStream = await FileSystem.OpenAppPackageFileAsync("products.json");
            using var reader = new StreamReader(jsonStream);
            var json = await reader.ReadToEndAsync();

            _products = JsonSerializer.Deserialize<Dictionary<string, Models.Product>>(json);
        }

        public List<string> Predict(Stream imageStream)
        {            
            using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imageStream);
            image.Mutate(x => x.Resize(416, 416));

            float[] input = new float[1 * 3 * 416 * 416];
            int idx = 0;
            
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < 416; y++)
                {
                    for (int x = 0; x < 416; x++)
                    {
                        var pixel = image[x, y];

                        float value = c switch
                        {
                            0 => pixel.R / 255f,
                            1 => pixel.G / 255f,
                            2 => pixel.B / 255f
                        };

                        input[idx++] = value;
                    }
                }
            }

            var tensor = new DenseTensor<float>(input, new[] { 1, 3, 416, 416 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", tensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();

            var dims = output.Dimensions;

            int numBoxes = dims[2];
            int numClasses = dims[1] - 4;

            List<(int classId, float conf)> detected = new();

            for (int i = 0; i < numBoxes; i++)
            {
                float maxConf = 0;
                int classId = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float conf = output[0, c + 4, i];

                    if (conf > maxConf)
                    {
                        maxConf = conf;
                        classId = c;
                    }
                }
                
                if (maxConf > 0.05f)
                {
                    detected.Add((classId, maxConf));
                }
            }
          
            var resultsList = detected
                .GroupBy(x => x.classId)
                .Select(g => g.OrderByDescending(x => x.conf).First())
                .Select(d =>
                {
                    int id = d.classId;

                    if (id < 0 || id >= classNames.Length)
                        return "Unknown";

                    var key = classNames[id];

                    if (_products != null && _products.ContainsKey(key))
                    {
                        var p = _products[key];
                        return $"Product: {p.name}     Code: {p.code}\n";
                    }

                    return key;
                })
                .ToList();

            return resultsList;
        }
    }
}