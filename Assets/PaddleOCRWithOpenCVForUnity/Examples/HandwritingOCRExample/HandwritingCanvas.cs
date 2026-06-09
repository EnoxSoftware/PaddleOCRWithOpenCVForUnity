using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace PaddleOCRWithOpenCVForUnityExample
{
    /// <summary>
    /// Handwriting UI: freehand drawing on a 640×640 <see cref="Mat"/>, mapping
    /// <see cref="TextureSelector"/> touch positions (POINT mode, Mat coordinates) via
    /// <see cref="Imgproc.line"/> / <see cref="Imgproc.circle"/>.
    /// </summary>
    /// <remarks>
    /// <b>Unity Editor setup</b>
    /// <list type="number">
    /// <item>Add Canvas and EventSystem to the scene.</item>
    /// <item>On the handwriting GameObject (e.g. HandwritingCanvas), add <see cref="RawImage"/> (RectTransform 640×640 recommended),
    /// <see cref="TextureSelector"/> (Selection Mode = POINT, Use OpenCV Mat Coordinates = on),
    /// and this component on the same object. A Canvas must exist in the parent hierarchy (TextureSelector requirement).
    /// Wire TextureSelector On Texture Selection State Changed to <see cref="OnTextureSelectionStateChanged"/>.
    /// The scene needs EventSystem and an Input Module (StandaloneInputModule or InputSystemUIInputModule).</item>
    /// <item>Keep the handwriting area on top or avoid overlap; UI in front of this RawImage cancels TextureSelector input.</item>
    /// </list>
    /// </remarks>
    [RequireComponent(typeof(RawImage))]
    [RequireComponent(typeof(TextureSelector))]
    public class HandwritingCanvas : MonoBehaviour
    {
        // Enums
        /// <summary>Stroke color.</summary>
        public enum StrokeColorKind
        {
            DarkGray,
            Red,
            Green,
            Blue,
            Orange,
            Purple,
            Black,
            White,
        }

        // Constants
        /// <summary>Canvas side length in pixels (easy to align with the OCR pipeline).</summary>
        public const int CANVAS_SIZE = 640;

        /// <summary>Minimum line thickness (integer passed to Imgproc.line).</summary>
        public const int MIN_LINE_THICKNESS = 1;

        /// <summary>Maximum line thickness (integer passed to Imgproc.line).</summary>
        public const int MAX_LINE_THICKNESS = 50;

        private static readonly Scalar BACKGROUND_COLOR = new Scalar(255, 255, 255, 255);

        // Public Fields
        [Tooltip("Line thickness (integer passed directly to Imgproc.line, 1–50).")]
        public int LineThickness = 3;

        /// <summary>Stroke color.</summary>
        public StrokeColorKind StrokeColor = StrokeColorKind.Black;

        [Tooltip("Stroke ended (right after the final line is written to the Mat). Wire HandwritingOCRExample.OnHandwritingStrokeEnded in the Inspector.")]
        public UnityEvent OnStrokeEnded = new UnityEvent();

        /// <summary>True while drawing a stroke with touch/mouse (until release).</summary>
        public bool IsPointerDown { get; private set; }

        // Private Fields
        private Mat _canvasMat;
        private Texture2D _texture;
        private RawImage _rawImage;
        private TextureSelector _textureSelector;
        private bool _hasLastPoint;
        private Point _lastPoint;

        // Unity Lifecycle Methods
        private void Awake()
        {
            _rawImage = GetComponent<RawImage>();
            _textureSelector = GetComponent<TextureSelector>();

            _canvasMat = new Mat(CANVAS_SIZE, CANVAS_SIZE, CvType.CV_8UC4, BACKGROUND_COLOR);

            _texture = new Texture2D(CANVAS_SIZE, CANVAS_SIZE, TextureFormat.RGBA32, false);
            OpenCVMatUtils.MatToTexture2D(_canvasMat, _texture);

            _rawImage.texture = _texture;
        }

        private void OnDestroy()
        {
            _canvasMat?.Dispose(); _canvasMat = null;

            if (_texture != null)
                Destroy(_texture);
            _texture = null;
        }

        // Public Methods
        /// <summary>
        /// Fills the Mat with white, resets stroke and TextureSelector selection state, and updates the texture.
        /// </summary>
        public void ClearCanvas()
        {
            if (_canvasMat == null || _texture == null)
                return;

            _canvasMat.setTo(BACKGROUND_COLOR);
            _hasLastPoint = false;
            IsPointerDown = false;
            _textureSelector?.ResetSelectionStatus();
            PushToTexture();
        }

        /// <summary>Mat holding current handwriting (for downstream OCR). Lifecycle matches this component.</summary>
        public Mat CanvasMat => _canvasMat;

        /// <summary>
        /// Called from <see cref="TextureSelector.OnTextureSelectionStateChanged"/> (wire in Inspector).
        /// </summary>
        public void OnTextureSelectionStateChanged(GameObject target, TextureSelector.TextureSelectionState state, Vector2[] points)
        {
            if (_canvasMat == null || _texture == null)
                return;

            if (state == TextureSelector.TextureSelectionState.POINT_SELECTION_CANCELLED)
            {
                CompleteStrokeIfPointerDown();
                _hasLastPoint = false;
                return;
            }

            if (state != TextureSelector.TextureSelectionState.POINT_SELECTION_STARTED &&
                state != TextureSelector.TextureSelectionState.POINT_SELECTION_IN_PROGRESS &&
                state != TextureSelector.TextureSelectionState.POINT_SELECTION_COMPLETED)
            {
                return;
            }

            Point raw = TextureSelector.ConvertSelectionPointsToOpenCVPoint(points);
            if (raw.x < 0 || raw.y < 0)
                return;

            Point current = ClampToCanvas(raw);

            int thickness = Mathf.Clamp(LineThickness, MIN_LINE_THICKNESS, MAX_LINE_THICKNESS);
            int dotRadius = thickness / 2;
            Scalar strokeColor = GetStrokeScalar();

            switch (state)
            {
                case TextureSelector.TextureSelectionState.POINT_SELECTION_STARTED:
                    IsPointerDown = true;
                    Imgproc.circle(_canvasMat, current, dotRadius, strokeColor, -1, Imgproc.LINE_AA, 0);
                    _lastPoint = current;
                    _hasLastPoint = true;
                    PushToTexture();
                    break;

                case TextureSelector.TextureSelectionState.POINT_SELECTION_IN_PROGRESS:
                    IsPointerDown = true;
                    if (_hasLastPoint)
                        Imgproc.line(_canvasMat, _lastPoint, current, strokeColor, thickness, Imgproc.LINE_AA, 0);
                    else
                        Imgproc.circle(_canvasMat, current, dotRadius, strokeColor, -1, Imgproc.LINE_AA, 0);
                    _lastPoint = current;
                    _hasLastPoint = true;
                    PushToTexture();
                    break;

                case TextureSelector.TextureSelectionState.POINT_SELECTION_COMPLETED:
                    if (_hasLastPoint)
                        Imgproc.line(_canvasMat, _lastPoint, current, strokeColor, thickness, Imgproc.LINE_AA, 0);
                    else
                        Imgproc.circle(_canvasMat, current, dotRadius, strokeColor, -1, Imgproc.LINE_AA, 0);
                    _lastPoint = current;
                    _hasLastPoint = true;
                    PushToTexture();
                    CompleteStrokeIfPointerDown();
                    break;
            }
        }

        // Private Methods
        /// <summary>
        /// Ends an in-progress stroke and notifies OCR via <see cref="OnStrokeEnded"/>.
        /// </summary>
        private void CompleteStrokeIfPointerDown()
        {
            if (!IsPointerDown)
                return;

            IsPointerDown = false;
            OnStrokeEnded?.Invoke();
        }

        private Point ClampToCanvas(Point p)
        {
            int maxX = _canvasMat.cols() - 1;
            int maxY = _canvasMat.rows() - 1;
            double x = Mathf.Clamp((float)p.x, 0f, maxX);
            double y = Mathf.Clamp((float)p.y, 0f, maxY);
            return new Point(x, y);
        }

        private void PushToTexture()
        {
            OpenCVMatUtils.MatToTexture2D(_canvasMat, _texture);
        }

        private Scalar GetStrokeScalar()
        {
            // CV_8UC4 Mat channel order is RGBA.
            switch (StrokeColor)
            {
                case StrokeColorKind.DarkGray:
                    return new Scalar(96, 96, 96, 255);
                case StrokeColorKind.Red:
                    return new Scalar(255, 0, 0, 255);
                case StrokeColorKind.Green:
                    return new Scalar(0, 128, 0, 255);
                case StrokeColorKind.Blue:
                    return new Scalar(0, 0, 255, 255);
                case StrokeColorKind.Orange:
                    return new Scalar(255, 128, 0, 255);
                case StrokeColorKind.Purple:
                    return new Scalar(128, 0, 128, 255);
                case StrokeColorKind.White:
                    return new Scalar(255, 255, 255, 255);
                default:
                    return new Scalar(0, 0, 0, 255);
            }
        }
    }
}
