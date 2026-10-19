import { PostList } from "@/components/landing/post-list";
import { getAllPosts, getPreferredBlogLang } from "@/core/blog";
import { getI18n } from "@/core/i18n/server";

import { useMDXComponents as getMDXComponents } from "../../../mdx-components";

// eslint-disable-next-line @typescript-eslint/unbound-method
const Wrapper = getMDXComponents().wrapper;

export const metadata = { title: "全部文章", filePath: "blog/index.mdx" };

export default async function PostsPage() {
  const { locale, t } = await getI18n();
  const localizedMetadata = { ...metadata, title: t.blog.allPosts };
  const posts = await getAllPosts(getPreferredBlogLang(locale));

  return (
    <Wrapper toc={[]} metadata={localizedMetadata} sourceCode="">
      <PostList title={localizedMetadata.title} posts={posts} />
    </Wrapper>
  );
}
