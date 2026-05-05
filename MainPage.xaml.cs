using FruitApp.Services;

namespace FruitApp
{
    public partial class MainPage : ContentPage
    {
        private OnnxService _onnx = new OnnxService();

        public MainPage()
        {
            InitializeComponent();
            Init();
        }

        private async void Init()
        {
            await _onnx.Init();
        }

        private async void OnTakePhoto(object sender, EventArgs e)
        {
            try
            {
                FileResult photo = null;
               
                if (MediaPicker.Default.IsCaptureSupported)
                {
                    try
                    {
                        photo = await MediaPicker.CapturePhotoAsync();
                    }
                    catch
                    {
                        // hvis kamera fejler (fx emulator), falder vi videre
                    }
                }
                
                if (photo == null)
                {
                    photo = await FilePicker.PickAsync(new PickOptions
                    {
                        PickerTitle = "Vælg billede",
                        FileTypes = FilePickerFileType.Images
                    });
                }

                if (photo == null)
                    return;

                using var stream = await photo.OpenReadAsync();
                
                var memory = new MemoryStream();
                await stream.CopyToAsync(memory);
                memory.Position = 0;

                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(memory.ToArray()));

                memory.Position = 0;

                var results = _onnx.Predict(memory);

                ResultLabel.Text = string.Join("\n", results);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", ex.Message, "OK");
            }
        }
    }
}

