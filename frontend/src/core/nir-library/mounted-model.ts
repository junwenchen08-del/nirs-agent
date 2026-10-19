export type MountedModel = {
  threadId: string;
  modelId: string;
  version: string;
  modelPath: string;
};

/** A composer selection is context, never evidence that inference has run. */
export function mountedModelContext(
  model: MountedModel | null | undefined,
  threadId: string,
  label: string,
): string {
  if (
    model?.threadId !== threadId ||
    !/^[A-Za-z0-9_-]{1,128}$/.test(model.modelId) ||
    !/^[A-Za-z0-9_-]{1,128}$/.test(model.version) ||
    model.modelPath !==
      `/mnt/user-data/outputs/models/${model.modelId}/${model.version}/model.pkl`
  )
    return "";
  return `${label}\n${JSON.stringify({ model_id: model.modelId, version: model.version, model_path: model.modelPath })}`;
}
