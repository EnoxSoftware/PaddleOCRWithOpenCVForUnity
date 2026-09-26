using System;
using System.Linq;
using OpenCVForUnity;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Extensions;
using OpenCVForUnity.Extensions.Runner;
using OpenCVForUnity.Extensions.SourceToMat;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.SourceToMat;
using OpenCVForUnity.UnityIntegration.Helper.UI;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using PaddleOCRWithOpenCVForUnity;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using OpenCVDebug = OpenCVForUnity.Extensions.OpenCVDebug;

namespace PaddleOCRWithOpenCVForUnityExample
{
    /// <summary>
    /// MultiSourceToMatHelper Example
    /// Switches between multiple input sources (webcam, video file, image file, or GPU readback) and processes each frame as an OpenCV <see cref="Mat"/>.
    ///
    /// Demonstrates:
    /// - Reading input frames via <see cref="MultiSourceToMatHelper"/> (webcam, video file, image file, or GPU readback)
    /// - Configuring <see cref="MultiSourceToMatHelper"/> per-kind settings via the helper Inspector and <see cref="SourceToMatControlPanel"/>
    /// - Runtime kind switching via the control panel MultiSection
    /// - Receiving frames via <see cref="SourceToMatHelperBase.OnFrameMatUpdated"/> (Inspector wiring)
    /// - Calling <see cref="SourceToMatHelperBase.Play"/> in OnInitialized only when not already playing or paused (<c>!IsPlaying &amp;&amp; !IsPaused</c>)
    /// - Recreating the preview texture on <see cref="SourceToMatHelperBase.OnFrameMatLayoutChanged"/>
    /// - Overlaying frame info with <see cref="Imgproc.putText"/> and measuring Mat update rate with <see cref="FPSCounter"/>
    /// - Displaying output via <see cref="Texture2D"/> or <see cref="RenderTexture"/>
    ///
    /// OpenCV classes and APIs used:
    /// - <see cref="Mat"/>, <see cref="Scalar"/>, <see cref="Point"/>
    /// - <see cref="Imgproc"/>: putText
    /// - <see cref="MultiSourceToMatHelper"/>, <see cref="MultiSourceHelperKind"/>, <see cref="SourceToMatColorFormat"/>
    /// - <see cref="OpenCVMatUnityUtils"/>: MatToTexture2D, MatToRenderTexture
    ///
    /// Unity integration:
    /// - The Mat returned by <see cref="SourceToMatHelperBase.FrameMat"/> is owned by the active helper; do not dispose it
    /// - <see cref="SourceToMatColorFormat.RGBA"/> matches Unity <see cref="TextureFormat.RGBA32"/> for display
    /// - WebGPU forces AsyncGPU inside WebCamTextureToMatHelper; this example does not switch helper types with <c>#if</c>
    /// - Reuse output textures across helper switches to avoid preview flicker
    /// - Helper lifecycle events (OnInitialized, OnFrameMatUpdated, etc.) are wired in the Inspector on <see cref="MultiSourceToMatHelper"/>
    /// - <see cref="SourceToMatControlPanel"/> on the same GameObject provides Transport / Transform / Kind / capability UI
    /// </summary>
    [RequireComponent(typeof(MultiSourceToMatHelper))]
    [RequireComponent(typeof(PaddleOCRComponent))]
    public class MultiSourceToMatHelperExample : MonoBehaviour
    {
        // Public Fields
        [Header("PaddleOCR")]
        [Tooltip("OCR inference component. Assign the PaddleOCRComponent on the same GameObject in the Inspector.")]
        public PaddleOCRComponent PaddleOCR;

        [Tooltip("TMP InputField for displaying recognition results.")]
        public TMP_InputField RecognitionResultField;

        // Public Fields
        [Header("Output")]
        /// <summary>
        /// The RawImage for previewing the result.
        /// </summary>
        public RawImage ResultPreview;

        [Space(10)]

        /// <summary>
        /// Whether RenderTexture is used when displaying rgbaMat in the scene; if Off, Texture2D is used.
        /// </summary>
        public Toggle OutputRenderTextureToggle;

        /// <summary>
        /// The cube rotated each frame to show the scene Update loop is running independently of Mat processing.
        /// </summary>
        public GameObject Cube;

        // Private Fields
        private Texture2D _outputTexture2D;
        private RenderTexture _outputRenderTexture;
        private GraphicsBuffer _graphicsBuffer;
        private MultiSourceToMatHelper _multiSourceToMatHelper;
        private FpsMonitor _fpsMonitor;
        private SourceToMatControlPanel _controlPanel;

        /// <summary>
        /// The FPS counter.
        /// Measure how frequently DidUpdateThisFrame() is actually updated.
        /// </summary>
        private FPSCounter _fpsCounter;

        /// <summary>
        /// The BGR Mat for PaddleOCR processing.
        /// </summary>
        private Mat _bgrMat;

        // Unity Lifecycle Methods
        private async void Start()
        {
            OpenCVDebug.SetDebugMode(true);

            _fpsMonitor = GetComponent<FpsMonitor>();

            // Get the MultiSourceToMatHelper component attached to the current game object.
            _multiSourceToMatHelper = GetComponent<MultiSourceToMatHelper>();

            if (PaddleOCR == null)
            {
                PaddleOCR = GetComponent<PaddleOCRComponent>();
            }

            if (PaddleOCR == null)
            {
                EngineLog.LogError($"{nameof(MultiSourceToMatHelperExample)}: {nameof(PaddleOCRComponent)} is not assigned.");
                return;
            }

            await PaddleOCR.WaitForInitializationAsync();
            if (!PaddleOCR.IsInitialized)
            {
                EngineLog.LogWarning($"{nameof(MultiSourceToMatHelperExample)}: {nameof(PaddleOCRComponent)} is not initialized.");
                return;
            }

            UpdateFpsMonitorInferenceInfo();

            // Per-kind paths and camera settings are configured on the helper Inspector and control panel.
            // RGBA matches Unity TextureFormat.RGBA32 used for RawImage preview.
            _multiSourceToMatHelper.OutputColorFormat = SourceToMatColorFormat.RGBA;

            if (OutputRenderTextureToggle != null && !SystemInfo.supportsComputeShaders)
            {
                OutputRenderTextureToggle.interactable = false;
            }

            // Subscribe to SourceToMatControlPanel events (Play/Pause/Stop, Rotate/Flip, HelperKind).
            WireSourceToMatControlPanelHooks();

            // Creates the active child helper and opens the requested input source.
            // OnSourceToMatHelperInitialized is raised when initialization completes (Inspector wiring).
            _multiSourceToMatHelper.Initialize();
        }

        private void Update()
        {
            // Count Mat updates while playing; MatUpdateFPS is shown on FpsMonitor and in the putText overlay.
            if (_fpsCounter != null
                && _multiSourceToMatHelper.IsInitialized
                && _multiSourceToMatHelper.IsPlaying
                && _multiSourceToMatHelper.DidUpdateThisFrame)
            {
                _fpsCounter.MeasureFPS();
            }

            if (_fpsMonitor != null && _fpsCounter != null)
            {
                _fpsMonitor.Add("MatUpdateFPS", _fpsCounter.GetCurrentFPS().ToString("F1"));
            }

            if (Cube != null)
            {
                Cube.transform.Rotate(new Vector3(0, 90, 0) * Time.deltaTime * 0.5f, Space.Self);
            }
        }

        private void OnDestroy()
        {
            // Unsubscribe control-panel listeners to avoid dangling callbacks.
            UnwireSourceToMatControlPanelHooks();

            PaddleOCR?.Cancel();
            _bgrMat?.Dispose(); _bgrMat = null;
            OpenCVDebug.SetDebugMode(false);
        }
        // Public Methods
        /// <summary>
        /// Raises the helper frame mat updated event.
        /// Draws overlays and updates the preview texture when a new frame is available during playback.
        /// </summary>
        public void OnSourceToMatHelperFrameMatUpdated()
        {
            // Invoked by the helper when a new frame is available (replaces Update + DidUpdateThisFrame for Mat processing).
            if (!_multiSourceToMatHelper.IsPlaying)
            {
                return;
            }

            // Returns the helper's internal Mat (RGBA); owned by the active child helper 窶・do not dispose.
            Mat rgbaMat = _multiSourceToMatHelper.FrameMat;
            if (rgbaMat == null)
            {
                return;
            }

            SubmitOCR(rgbaMat);

            // Draw helper-kind label and frame stats in-place before display conversion.
            Imgproc.putText(
                rgbaMat,
                GetHelperKindOverlayText(),
                new Point(5, 30),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7,
                new Scalar(255, 255, 255, 255),
                2,
                Imgproc.LINE_AA,
                false);

            Imgproc.putText(
                rgbaMat,
                "W:" + rgbaMat.width() + " H:" + rgbaMat.height() + " SO:" + Screen.orientation
                + " MatUpdateFPS:" + (_fpsCounter != null ? _fpsCounter.GetCurrentFPS() : 0f),
                new Point(5, rgbaMat.rows() - 10),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7,
                new Scalar(255, 255, 255, 255),
                2,
                Imgproc.LINE_AA,
                false);

            UpdatePreviewFromFrameMat(rgbaMat);
        }

        /// <summary>
        /// Raises the helper initialized event.
        /// Recreates the preview texture and starts playback on first initialization.
        /// Skips Play when re-initialization has already restored Playing or Paused.
        /// </summary>
        public void OnSourceToMatHelperInitialized()
        {
            EngineLog.Log("OnSourceToMatHelperInitialized");

            // Retain output textures across helper switches to prevent RawImage flicker.
            ReleasePreviewResources();
            RecreatePreviewTexture();
            EnsureBgrMat(_multiSourceToMatHelper.FrameMat);

            if (_fpsMonitor != null)
            {
                //_fpsMonitor.Clear();
                UpdateFpsMonitorPlaybackState();
                UpdateFpsMonitorSourceInfo();
            }

            _fpsCounter = new FPSCounter(1.0f);

            // Call Play only when the helper is not already playing or paused.
            // Re-initialization (e.g. kind switch or RenderTexture toggle) may restore the previous playback state.
            if (!_multiSourceToMatHelper.IsPlaying && !_multiSourceToMatHelper.IsPaused)
            {
                _multiSourceToMatHelper.Play();
                UpdateFpsMonitorPlaybackState();
            }
        }

        /// <summary>
        /// Raises the helper frame mat layout changed event.
        /// Recreates the preview texture when rotation or output size changes.
        /// </summary>
        public void OnSourceToMatHelperFrameMatLayoutChanged()
        {
            EngineLog.Log("OnSourceToMatHelperFrameMatLayoutChanged");

            // Raised when Rotate90 or output size changes; recreate the preview texture to match FrameMat layout.
            _bgrMat?.Dispose(); _bgrMat = null;
            RecreatePreviewTexture();
            EnsureBgrMat(_multiSourceToMatHelper.FrameMat);

            if (_fpsMonitor != null)
            {
                _fpsMonitor.Add("Width", _multiSourceToMatHelper.Width.ToString());
                _fpsMonitor.Add("Height", _multiSourceToMatHelper.Height.ToString());
                _fpsMonitor.Add("Orientation", Screen.orientation.ToString());
            }
        }

        /// <summary>
        /// Raises the helper released event.
        /// </summary>
        public void OnSourceToMatHelperReleased()
        {
            EngineLog.Log("OnSourceToMatHelperReleased");

            if (_fpsMonitor != null)
            {
                _fpsMonitor.Clear();
                UpdateFpsMonitorInferenceInfo();
            }

            _fpsCounter = null;

            PaddleOCR?.Cancel();
            _bgrMat?.Dispose(); _bgrMat = null;

            // Destroy preview textures; the helper retains ownership of FrameMat.
            CleanupPreviewResources();
        }

        /// <summary>
        /// Raises the helper disposed event.
        /// </summary>
        public void OnSourceToMatHelperDisposed()
        {
            EngineLog.Log("OnSourceToMatHelperDisposed");

            PaddleOCR?.Cancel();
            _bgrMat?.Dispose(); _bgrMat = null;

            // Destroy preview textures and set references to null.
            CleanupPreviewResources();
        }

        /// <summary>
        /// Raises the helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        /// <param name="message">Message.</param>
        public void OnSourceToMatHelperErrorOccurred(SourceToMatErrorCode errorCode, string message)
        {
            EngineLog.Log("OnSourceToMatHelperErrorOccurred " + errorCode + ":" + message);

            if (_fpsMonitor != null)
            {
                _fpsMonitor.ConsoleText = "ErrorCode: " + errorCode + ":" + message;
            }
        }

        /// <summary>
        /// Handles PaddleOCR inference completion and updates the recognition result field.
        /// </summary>
        /// <param name="kind">Completion kind.</param>
        /// <param name="errorMessage">Message on failure or cancellation.</param>
        public void OnOCRWorkCompleted(WorkCompletionKind kind, string errorMessage)
        {
            if (kind != WorkCompletionKind.Succeeded)
            {
                if (kind == WorkCompletionKind.Faulted && !string.IsNullOrEmpty(errorMessage))
                {
                    EngineLog.LogWarning($"{nameof(MultiSourceToMatHelperExample)} OCR: {errorMessage}");
                }

                return;
            }

            UpdateRecognitionResultField();
        }

        /// <summary>
        /// Updates inference information after PaddleOCR settings change.
        /// </summary>
        public void OnPaddleOCRInferenceSettingsChanged()
        {
            UpdateFpsMonitorInferenceInfo();
        }

        /// <summary>
        /// Raises the back button click event.
        /// Stops playback and disposes the helper before scene transition (required on WebGL).
        /// </summary>
        public async void OnBackButtonClick()
        {
            // Stop and dispose before leaving the scene (required on WebGL to release the active source).
            if (_multiSourceToMatHelper.IsPlaying || _multiSourceToMatHelper.IsPaused)
            {
                await _multiSourceToMatHelper.StopAsync();
            }

            await _multiSourceToMatHelper.DisposeAsync();

            // Load the main menu scene when the back button is clicked.
            SceneManager.LoadScene("PaddleOCRWithOpenCVForUnityExample");
        }

        /// <summary>
        /// Invoked by <see cref="SourceToMatControlPanel.OnAfterPlay"/>.
        /// </summary>
        public void OnControlPanelAfterPlay()
        {
            UpdateFpsMonitorPlaybackState();
        }

        /// <summary>
        /// Invoked by <see cref="SourceToMatControlPanel.OnAfterPause"/>.
        /// </summary>
        public void OnControlPanelAfterPause()
        {
            UpdateFpsMonitorPlaybackState();
        }

        /// <summary>
        /// Invoked by <see cref="SourceToMatControlPanel.OnAfterStop"/>.
        /// </summary>
        public void OnControlPanelAfterStop()
        {
            UpdateFpsMonitorPlaybackState();
        }

        /// <summary>
        /// Invoked by <see cref="SourceToMatControlPanel.OnRotate90Changed"/>.
        /// </summary>
        /// <param name="isOn">New Rotate90Degree value applied by the panel.</param>
        public void OnControlPanelRotate90Changed(bool isOn)
        {
            if (_fpsMonitor != null)
            {
                _fpsMonitor.Add("Rotate90Degree", isOn.ToString());
            }
        }

        /// <summary>
        /// Invoked by <see cref="SourceToMatControlPanel.OnFlipVerticalChanged"/>.
        /// </summary>
        /// <param name="isOn">New FlipVertical value applied by the panel.</param>
        public void OnControlPanelFlipVerticalChanged(bool isOn)
        {
            if (_fpsMonitor != null)
            {
                _fpsMonitor.Add("FlipVertical", isOn.ToString());
            }
        }

        /// <summary>
        /// Invoked by <see cref="SourceToMatControlPanel.OnFlipHorizontalChanged"/>.
        /// </summary>
        /// <param name="isOn">New FlipHorizontal value applied by the panel.</param>
        public void OnControlPanelFlipHorizontalChanged(bool isOn)
        {
            if (_fpsMonitor != null)
            {
                _fpsMonitor.Add("FlipHorizontal", isOn.ToString());
            }
        }

        /// <summary>
        /// Invoked by <see cref="SourceToMatControlPanel.OnHelperKindChanged"/>.
        /// </summary>
        /// <param name="kindIndex">Dropdown index matching <see cref="MultiSourceHelperKind"/>.</param>
        public void OnControlPanelHelperKindChanged(int kindIndex)
        {
            if (_fpsMonitor == null || !Enum.IsDefined(typeof(MultiSourceHelperKind), kindIndex))
            {
                return;
            }

            _fpsMonitor.Add("HelperKind", ((MultiSourceHelperKind)kindIndex).ToString());
            _fpsMonitor.Add("ActiveHelper", GetActiveHelperTypeName());
        }

        /// <summary>
        /// Raises the output RenderTexture toggle value changed event.
        /// Re-initializes the helper so the preview path switches between Texture2D and RenderTexture.
        /// </summary>
        public void OnOutputRenderTextureToggleValueChanged()
        {
            if (_multiSourceToMatHelper.IsInitialized)
            {
                _multiSourceToMatHelper.Initialize();
            }
        }

        // Private Methods
        private void SubmitOCR(Mat rgbaMat)
        {
            if (PaddleOCR == null || !PaddleOCR.IsInitialized)
            {
                return;
            }

            EnsureBgrMat(rgbaMat);
            if (_bgrMat == null)
            {
                return;
            }

            Imgproc.cvtColor(rgbaMat, _bgrMat, Imgproc.COLOR_RGBA2BGR);
            PaddleOCR.Submit(_bgrMat);

            if (PaddleOCR.TryGetLatestResultViews(out Mat[] detView, out Mat[] clsView, out Mat[] recView))
            {
                PaddleOCRPipelineUtility.VisualizeOCRResults(
                    rgbaMat,
                    detView,
                    clsView,
                    recView,
                    printResult: true,
                    isRGB: true);
            }
        }

        private void EnsureBgrMat(Mat rgbaMat)
        {
            if (rgbaMat == null || rgbaMat.empty())
            {
                return;
            }

            if (_bgrMat != null
                && (_bgrMat.rows() != rgbaMat.rows() || _bgrMat.cols() != rgbaMat.cols()))
            {
                _bgrMat.Dispose();
                _bgrMat = null;
            }

            if (_bgrMat == null)
            {
                _bgrMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
            }
        }

        private void UpdateRecognitionResultField()
        {
            if (RecognitionResultField == null || PaddleOCR == null)
            {
                return;
            }

            if (!PaddleOCR.TryGetLatestParsedResult(out PaddleOCRParsedResult parsed))
            {
                return;
            }

            var recognitions = parsed.Recognitions;
            string text = recognitions == null || recognitions.Count == 0
                ? string.Empty
                : string.Join("\n", recognitions.Select(r => r.text));

            RecognitionResultField.text = text;
            RecognitionResultField.stringPosition = 0;
            RecognitionResultField.caretPosition = 0;
            if (RecognitionResultField.verticalScrollbar != null)
            {
                RecognitionResultField.verticalScrollbar.value = 0;
            }
        }

        private void UpdateFpsMonitorInferenceInfo()
        {
            if (_fpsMonitor == null || PaddleOCR == null)
            {
                return;
            }

            _fpsMonitor.Add(
                "inferenceFramework",
                InferenceFrameworkUtils.GetSelectionDisplayName(PaddleOCR.InferenceFramework));

            TextDetectorMultiBackend detector = PaddleOCR.Detector;
            if (detector != null)
            {
                int backend = detector.DnnBackend;
                int target = detector.DnnTarget;
                _fpsMonitor.Add("dnnBackend", MultiBackendNet.GetBackendDisplayString(backend));
                _fpsMonitor.Add("dnnTarget", MultiBackendNet.GetTargetDisplayString(target));
            }
            else
            {
                _fpsMonitor.Add("dnnBackend", "-");
                _fpsMonitor.Add("dnnTarget", "-");
            }

            _fpsMonitor.Add("useAsyncInference", PaddleOCR.UseAsyncInference.ToString());
        }

        private void RecreatePreviewTexture()
        {
            Mat rgbaMat = _multiSourceToMatHelper.FrameMat;
            if (rgbaMat == null)
            {
                return;
            }

            ReleasePreviewResources();

            if (OutputRenderTextureToggle == null || !OutputRenderTextureToggle.isOn)
            {
                // Texture dimensions must match Mat cols()/rows() (width/height may swap when rotated).
                _outputTexture2D = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);

                // CPU path: convert the Mat to a Texture2D for RawImage preview.
                OpenCVMatUnityUtils.MatToTexture2D(rgbaMat, _outputTexture2D);
                ApplyPreviewTexture(_outputTexture2D);
                return;
            }

            _graphicsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)rgbaMat.total(), (int)rgbaMat.elemSize());
            _outputRenderTexture = new RenderTexture(rgbaMat.width(), rgbaMat.height(), 0);
            _outputRenderTexture.enableRandomWrite = true;
            _outputRenderTexture.Create();

            try
            {
                // GPU path: upload Mat data into a RenderTexture via GraphicsBuffer.
                OpenCVMatUnityUtils.MatToRenderTexture(rgbaMat, _outputRenderTexture, _graphicsBuffer);
            }
            catch (Exception ex)
            {
                if (_fpsMonitor != null)
                {
                    _fpsMonitor.ConsoleText = ex.Message;
                }
            }

            ApplyPreviewTexture(_outputRenderTexture);
        }

        private void CleanupPreviewResources()
        {
            ReleasePreviewResources();

            UpdateFpsMonitorPlaybackState();
        }

        private void ReleasePreviewResources()
        {
            if (_outputTexture2D != null)
            {
                Texture2D.Destroy(_outputTexture2D);
                _outputTexture2D = null;
            }

            if (_outputRenderTexture != null)
            {
                RenderTexture.Destroy(_outputRenderTexture);
                _outputRenderTexture = null;
            }

            if (_graphicsBuffer != null)
            {
                _graphicsBuffer.Dispose();
                _graphicsBuffer = null;
            }
        }

        private void UpdateFpsMonitorPlaybackState()
        {
            if (_fpsMonitor == null)
            {
                return;
            }

            _fpsMonitor.Add("PlaybackState", GetPlaybackStateText());
        }

        private string GetPlaybackStateText()
        {
            if (!_multiSourceToMatHelper.IsInitialized)
            {
                return "Uninitialized";
            }

            if (_multiSourceToMatHelper.IsPlaying)
            {
                return "Playing";
            }

            if (_multiSourceToMatHelper.IsPaused)
            {
                return "Paused";
            }

            return "Ready";
        }

        private void WireSourceToMatControlPanelHooks()
        {
            // Runtime wiring for control-panel callbacks (Inspector does not wire these).
            _controlPanel = GetComponent<SourceToMatControlPanel>();
            if (_controlPanel == null)
            {
                return;
            }

            _controlPanel.OnAfterPlay.AddListener(OnControlPanelAfterPlay);
            _controlPanel.OnAfterPause.AddListener(OnControlPanelAfterPause);
            _controlPanel.OnAfterStop.AddListener(OnControlPanelAfterStop);
            _controlPanel.OnRotate90Changed.AddListener(OnControlPanelRotate90Changed);
            _controlPanel.OnFlipVerticalChanged.AddListener(OnControlPanelFlipVerticalChanged);
            _controlPanel.OnFlipHorizontalChanged.AddListener(OnControlPanelFlipHorizontalChanged);
            _controlPanel.OnHelperKindChanged.AddListener(OnControlPanelHelperKindChanged);
        }

        private void UnwireSourceToMatControlPanelHooks()
        {
            if (_controlPanel == null)
            {
                return;
            }

            _controlPanel.OnAfterPlay.RemoveListener(OnControlPanelAfterPlay);
            _controlPanel.OnAfterPause.RemoveListener(OnControlPanelAfterPause);
            _controlPanel.OnAfterStop.RemoveListener(OnControlPanelAfterStop);
            _controlPanel.OnRotate90Changed.RemoveListener(OnControlPanelRotate90Changed);
            _controlPanel.OnFlipVerticalChanged.RemoveListener(OnControlPanelFlipVerticalChanged);
            _controlPanel.OnFlipHorizontalChanged.RemoveListener(OnControlPanelFlipHorizontalChanged);
            _controlPanel.OnHelperKindChanged.RemoveListener(OnControlPanelHelperKindChanged);
            _controlPanel = null;
        }

        private void UpdatePreviewFromFrameMat(Mat rgbaMat)
        {
            if (OutputRenderTextureToggle == null || !OutputRenderTextureToggle.isOn)
            {
                if (_outputTexture2D != null)
                {
                    // CPU path: copy Mat into a reusable Texture2D for RawImage.
                    OpenCVMatUnityUtils.MatToTexture2D(rgbaMat, _outputTexture2D);
                }

                return;
            }

            if (_outputRenderTexture != null && _graphicsBuffer != null)
            {
                // GPU path: upload Mat data into a reusable RenderTexture.
                OpenCVMatUnityUtils.MatToRenderTexture(rgbaMat, _outputRenderTexture, _graphicsBuffer);
            }
        }

        private void ApplyPreviewTexture(Texture texture)
        {
            if (ResultPreview == null)
            {
                return;
            }

            // Set the texture as the RawImage preview and keep aspect ratio in sync.
            ResultPreview.texture = texture;
            AspectRatioFitter aspectRatioFitter = ResultPreview.GetComponent<AspectRatioFitter>();
            if (aspectRatioFitter != null)
            {
                aspectRatioFitter.aspectRatio = (float)texture.width / texture.height;
            }
        }

        private void UpdateFpsMonitorSourceInfo()
        {
            _fpsMonitor.Add("HelperKind", _multiSourceToMatHelper.RequestedHelperKind.ToString());
            _fpsMonitor.Add("ActiveHelper", GetActiveHelperTypeName());
            _fpsMonitor.Add("Width", _multiSourceToMatHelper.Width.ToString());
            _fpsMonitor.Add("Height", _multiSourceToMatHelper.Height.ToString());
            _fpsMonitor.Add("Orientation", Screen.orientation.ToString());
            _fpsMonitor.Add("Rotate90Degree", _multiSourceToMatHelper.Rotate90Degree.ToString());
            _fpsMonitor.Add("FlipVertical", _multiSourceToMatHelper.FlipVertical.ToString());
            _fpsMonitor.Add("FlipHorizontal", _multiSourceToMatHelper.FlipHorizontal.ToString());

            IMatSource matSource = _multiSourceToMatHelper.MatSource;
            if (matSource is ICameraMatSource camera)
            {
                _fpsMonitor.Add("DeviceName", camera.DeviceName);
                _fpsMonitor.Add("FPS", camera.FPS.ToString());
            }

            if (matSource is ICameraFacingControllable facing)
            {
                _fpsMonitor.Add("IsFrontFacing", facing.IsFrontFacing.ToString());
            }

            if (matSource is IVideoFileMatSource video)
            {
                _fpsMonitor.Add("VideoPath", video.RequestedVideoFilePath);
                _fpsMonitor.Add("VideoFPS", video.FPS.ToString());
                _fpsMonitor.Add("Loop", video.Loop.ToString());
            }

            if (matSource is IImageFileMatSource image)
            {
                _fpsMonitor.Add("ImagePath", image.RequestedImageFilePath);
                _fpsMonitor.Add("Repeat", image.Repeat.ToString());
            }

#if !OPENCV_DONT_USE_WEBCAMTEXTURE_API
            if (_multiSourceToMatHelper.ActiveHelper is WebCamTextureToMatHelper webCamHelper)
            {
                _fpsMonitor.Add("RequestedUseAsyncGPUReadback", webCamHelper.RequestedUseAsyncGPUReadback.ToString());
                _fpsMonitor.Add("EffectiveUseAsyncGPUReadback", webCamHelper.EffectiveUseAsyncGPUReadback.ToString());
            }
#endif
        }

        private string GetActiveHelperTypeName()
        {
            SourceToMatHelperBase activeHelper = _multiSourceToMatHelper.ActiveHelper;
            return activeHelper != null ? activeHelper.GetType().Name : "(none)";
        }

        private string GetHelperKindOverlayText()
        {
            switch (_multiSourceToMatHelper.RequestedHelperKind)
            {
                case MultiSourceHelperKind.WebCamTexture:
#if !OPENCV_DONT_USE_WEBCAMTEXTURE_API
                    if (_multiSourceToMatHelper.ActiveHelper is WebCamTextureToMatHelper webCamHelper
                        && webCamHelper.EffectiveUseAsyncGPUReadback)
                    {
                        return "WebCamTexture -> RenderTexture => Mat";
                    }
#endif
                    return "WebCamTexture => Mat";
                case MultiSourceHelperKind.VideoCapture:
                case MultiSourceHelperKind.UnityVideoPlayer:
                    return "Video File => Mat";
                case MultiSourceHelperKind.ImageFile:
                    return "Image File => Mat";
                case MultiSourceHelperKind.AsyncGPUReadback:
                    return "Camera => RenderTexture => Mat";
                default:
                    return _multiSourceToMatHelper.RequestedHelperKind.ToString() + " => Mat";
            }
        }
    }
}
