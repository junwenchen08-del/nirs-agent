"use client";

import {
  BookOpenIcon,
  BoxesIcon,
  DatabaseIcon,
  MessagesSquare,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import {
  SidebarGroup,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useI18n } from "@/core/i18n/hooks";
import { useNirLibraryEnabled } from "@/core/nir-library";

export function WorkspaceNavChatList() {
  const { t } = useI18n();
  const pathname = usePathname();
  const { enabled: nirLibraryEnabled } = useNirLibraryEnabled();
  return (
    <SidebarGroup className="pt-1">
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton isActive={pathname === "/workspace/chats"} asChild>
            <Link className="text-muted-foreground" href="/workspace/chats">
              <MessagesSquare />
              <span>{t.sidebar.chats}</span>
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
        {nirLibraryEnabled && (
          <>
            <SidebarMenuItem>
              <SidebarMenuButton
                isActive={pathname.startsWith("/workspace/nir/datasets")}
                asChild
              >
                <Link
                  className="text-muted-foreground"
                  href="/workspace/nir/datasets"
                >
                  <DatabaseIcon />
                  <span>{t.sidebar.nirDatasets}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton
                isActive={pathname.startsWith("/workspace/nir/models")}
                asChild
              >
                <Link
                  className="text-muted-foreground"
                  href="/workspace/nir/models"
                >
                  <BoxesIcon />
                  <span>{t.sidebar.nirModels}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </>
        )}
        <SidebarMenuItem>
          <SidebarMenuButton
            isActive={pathname.startsWith("/workspace/methods")}
            asChild
          >
            <Link className="text-muted-foreground" href="/workspace/methods">
              <BookOpenIcon />
              <span>{t.sidebar.methods}</span>
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroup>
  );
}
