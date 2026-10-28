using System;
using System.IO;
using Verse;

namespace DiffusionMaps
{
    public class DiffusionMapsMod : Mod
    {
        public static DiffusionMapsMod Instance { get; private set; }

        public const string ModelName = "v1_mdm";

        public DiffusionMapsMod(ModContentPack content) : base(content)
        {
            Instance = this;
        }

        public string ModelDir => Path.Combine(Content.RootDir, "Models", ModelName);
        public string NativeDir => Path.Combine(Content.RootDir, "Native", "win-x64");

        static LayoutModel model;

        /// <summary>
        /// The shared model, loaded on first use (about a second). Throws with a readable message if ONNX Runtime
        /// or the model files cannot be loaded.
        /// </summary>
        public static LayoutModel Model
        {
            get
            {
                if (model == null)
                    model = Instance.LoadModel(threads: 0);
                return model;
            }
        }

        public LayoutModel LoadModel(int threads)
        {
            OnnxNative.EnsureLoaded(NativeDir);
            return new LayoutModel(Path.Combine(ModelDir, "model.onnx"), Path.Combine(ModelDir, "model_meta.json"), threads);
        }
    }
}
