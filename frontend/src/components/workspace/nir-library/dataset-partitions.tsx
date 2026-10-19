import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  datasetChatPath,
  datasetVersionPath,
  validPartition,
} from "@/core/nir-library/dataset-partitions";
import type {
  DatasetPartition,
  NIRDatasetProfile,
} from "@/core/nir-library/types";

export function PartitionCounts({
  partition,
  zh,
}: {
  partition: DatasetPartition;
  zh: boolean;
}) {
  return (
    <p className="text-muted-foreground text-sm">
      {zh ? "训练集" : "Training"} {partition.counts.calibration} ·{" "}
      {zh ? "验证集" : "Validation"} {partition.counts.validation} ·{" "}
      {zh ? "测试集" : "Test"} {partition.counts.test}
    </p>
  );
}

export function DatasetPartitions({
  datasetId,
  profiles,
  zh,
}: {
  datasetId: string;
  profiles: NIRDatasetProfile[];
  zh: boolean;
}) {
  const splits = profiles.filter(
    (profile) =>
      profile.dataset_id === datasetId &&
      profile.profile_status === "split" &&
      profile.partition &&
      validPartition(profile.partition, profile.id),
  );
  return (
    <section
      className="space-y-3 rounded-xl border p-4"
      aria-label={zh ? "已保存的划分版本" : "Saved partitions"}
    >
      <h3 className="font-medium">
        {zh ? "已保存的划分版本" : "Saved partitions"}
      </h3>
      <p className="text-muted-foreground text-sm">
        {zh
          ? "每个版本固定训练、验证和测试样本，保留原始数据与建模时的样本顺序。"
          : "Each version preserves training, validation and test membership and the sample order used for modeling."}
      </p>
      {splits.length === 0 ? (
        <p className="text-muted-foreground text-sm">
          {zh
            ? "暂未保存划分版本。新建模完成数据划分后会自动保存。"
            : "No saved partitions. New modeling sessions save their partition after ingestion."}
        </p>
      ) : (
        splits.map((profile) => (
          <div
            key={profile.id}
            className="flex flex-wrap items-center justify-between gap-3 rounded-lg border p-3"
          >
            <div className="space-y-1">
              <p className="text-sm font-medium">
                {zh ? "划分版本" : "Partition"} v{profile.profile_version} ·{" "}
                {profile.partition!.method === "random"
                  ? zh
                    ? "自动划分"
                    : "Automatic split"
                  : zh
                    ? "已有固定划分"
                    : "Fixed input split"}
              </p>
              <PartitionCounts partition={profile.partition!} zh={zh} />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button asChild size="sm" variant="outline">
                <Link href={datasetVersionPath(datasetId, profile.id)}>
                  {zh ? "查看划分" : "View partition"}
                </Link>
              </Button>
              <Button asChild size="sm">
                <Link href={datasetChatPath(datasetId, profile.id)}>
                  {zh ? "复用此划分" : "Reuse partition"}
                </Link>
              </Button>
            </div>
          </div>
        ))
      )}
    </section>
  );
}
